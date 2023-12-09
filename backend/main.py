from contextlib import asynccontextmanager
from io import StringIO
import numpy as np
from prisma import Prisma
from fastapi.middleware.gzip import GZipMiddleware
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from interceptors.base_response import IBaseResponse
from http import HTTPStatus
import pandas as pd
import json
import re
import textdistance
from sklearn.cluster import AgglomerativeClustering
import scipy.stats as ss
import uuid
import os

# Init server
# uvicorn backend.main:app --reload

prisma = Prisma()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await prisma.connect()
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.get("/healthcheck")
def get_healthcheck():
    return "OK"


@app.post("/upload")
async def upload_file(request: Request):
    try:
        csv = pd.read_json(StringIO(await request.json()))
        if not os.path.exists("upload"):
            os.makedirs("upload")
        csv.to_csv("upload/data.csv")

        return IBaseResponse(
            statusCode=HTTPStatus.OK, message="Uploaded file to backend successfully"
        )
    except Exception as e:
        return IBaseResponse(
            statusCode=HTTPStatus.INTERNAL_SERVER_ERROR, message="Error", error=str(e)
        )


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@app.post("/process-dirty-data")
async def process_dirty_data(request: Request):
    # read file from local
    df = pd.read_csv("upload/data.csv")

    # process data
    payload = {
        "distinct_items_count": len(pd.unique(df["Item_Identifier"])),
        "mean_item_weight": round(float(df["Item_Weight"].mean()), 2),
        "item_fat_content_types": df.fillna("Unknown")
        .groupby("Item_Fat_Content")["Item_Identifier"]
        .nunique()
        .to_dict(),
        "mean_item_visibility": round(float((df["Item_Visibility"].mean()) * 100), 2),
        "mean_item_mrp": round(float(df["Item_MRP"].mean()), 2),
        "distinct_outlets_count": len(pd.unique(df["Outlet_Identifier"])),
        "oldest_outlet_establishment_year": df["Outlet_Establishment_Year"].min(),
        "newest_outlet_establishment_year": df["Outlet_Establishment_Year"].max(),
        "outlet_size_types": df.fillna("Unknown")
        .groupby("Outlet_Size")["Outlet_Identifier"]
        .nunique()
        .to_dict(),
        "outlet_location_types": df.fillna("Unknown")
        .groupby("Outlet_Location_Type")["Outlet_Identifier"]
        .nunique()
        .to_dict(),
        "outlet_types": df.fillna("Unknown")
        .groupby("Outlet_Type")["Outlet_Identifier"]
        .nunique()
        .to_dict(),
        "mean_item_outlet_sales_value": df["Item_Outlet_Sales"].mean(),
    }

    # Encode numpy data to prevent JSON serialization error
    payload_processed = json.dumps(payload, cls=NpEncoder)

    return IBaseResponse(
        statusCode=HTTPStatus.OK,
        message="Dirty data processed successfully",
        data=payload_processed,
    )


def normalize_text(texts: list, threshold: float = 0.4):
    def normalize(value):
        """Keep only lower-cased value and numbers"""
        return re.sub("[^a-z0-9]+", " ", value.lower())

    def group_values(values):
        """Replace each value with the representative of its cluster"""
        normalized_values = np.array([normalize(value) for value in values])
        distances = 1 - np.array(
            [
                [textdistance.jaro_winkler(one, another) for one in normalized_values]
                for another in normalized_values
            ]
        )
        clustering = AgglomerativeClustering(
            distance_threshold=threshold,  # this parameter needs to be tuned carefully
            metric="precomputed",
            linkage="complete",
            n_clusters=None,
        ).fit(distances)
        centers = dict()
        for cluster_id in set(clustering.labels_):
            index = clustering.labels_ == cluster_id
            centrality = distances[:, index][index].sum(axis=1)
            centers[cluster_id] = normalized_values[index][centrality.argmin()]
        return [centers[i] for i in clustering.labels_]

    return group_values(texts)


def unordered_categorical_correlation(df: pd.DataFrame, columns: list[str]):
    """Compute correlation between unordered categorical variables (strings only)"""

    def cramers_corrected_stat(columnA: pd.Series, columnB: pd.Series):
        """
        Calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328
        """
        confusion_matrix = pd.crosstab(columnA, columnB)
        chi2 = ss.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.to_numpy().sum(axis=None)
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)

        return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

    def compute_category_correlation(df: pd.DataFrame, columns: list[str]):
        """Compute the correlation between string columns of a DataFrame"""
        new_columns = [f"{column}_cat_code" for column in columns]
        for column in columns:
            df[f"{column}_cat_code"] = df[column].astype("category").cat.codes
        result = df[new_columns].corr(method=cramers_corrected_stat)  # type: ignore

        return result

    return compute_category_correlation(df, columns)


async def create_outlets_in_db(df: pd.DataFrame):
    # create entries in db for outlet_size_types
    outlet_size_types = df["Outlet_Size"].unique().tolist()
    await prisma.outlet_size_types.delete_many()
    outlet_size_types_ids = [
        {"id": str(uuid.uuid4()), "name": outlet_size_type}
        for outlet_size_type in outlet_size_types
    ]
    await prisma.outlet_size_types.create_many(outlet_size_types_ids)  # type: ignore

    # replace outlet_size with id
    df["Outlet_Size"] = df["Outlet_Size"].replace(
        to_replace=outlet_size_types,
        value=[item["id"] for item in outlet_size_types_ids],
    )

    # create entries in db for outlet_location_types
    outlet_location_types = df["Outlet_Location_Type"].unique().tolist()
    await prisma.outlet_location_types.delete_many()
    outlet_location_types_ids = [
        {"id": str(uuid.uuid4()), "name": outlet_location_type}
        for outlet_location_type in outlet_location_types
    ]
    await prisma.outlet_location_types.create_many(
        outlet_location_types_ids  # type: ignore
    )

    # replace outlet_location_type with id
    df["Outlet_Location_Type"] = df["Outlet_Location_Type"].replace(
        to_replace=outlet_location_types,
        value=[item["id"] for item in outlet_location_types_ids],
    )

    # create entries in db for outlet_types
    outlet_types = df["Outlet_Type"].unique().tolist()
    await prisma.outlet_types.delete_many()
    outlet_types_ids = [
        {"id": str(uuid.uuid4()), "name": outlet_type} for outlet_type in outlet_types
    ]
    await prisma.outlet_types.create_many(outlet_types_ids)  # type: ignore

    # replace outlet_type with id
    df["Outlet_Type"] = df["Outlet_Type"].replace(
        to_replace=outlet_types, value=[item["id"] for item in outlet_types_ids]
    )

    # rename columns
    df = df.rename(
        columns={
            "Outlet_Identifier": "id",
            "Outlet_Establishment_Year": "establishment_year",
            "Outlet_Size": "outlet_size_type",
            "Outlet_Location_Type": "outlet_location_type",
            "Outlet_Type": "outlet_type",
        }
    )

    # create entries in db for outlets
    outlets = df[
        [
            "id",
            "establishment_year",
            "outlet_size_type",
            "outlet_location_type",
            "outlet_type",
        ]
    ].drop_duplicates()
    outlets = outlets.to_dict("records")
    await prisma.outlets.delete_many()
    await prisma.outlets.create_many(outlets)  # type: ignore


async def create_items_in_db(df: pd.DataFrame):
    # create entries in db for item_fat_content
    item_fat_contents = df["Item_Fat_Content"].unique().tolist()
    await prisma.fat_content_types.delete_many()  # delete all existing entries first
    item_fat_contents_ids = [
        {"id": str(uuid.uuid4()), "name": item_fat_content}
        for item_fat_content in item_fat_contents
    ]
    await prisma.fat_content_types.create_many(item_fat_contents_ids)  # type: ignore

    # replace item_fat_content with id
    df["Item_Fat_Content"] = df["Item_Fat_Content"].replace(
        to_replace=item_fat_contents,
        value=[item["id"] for item in item_fat_contents_ids],
    )

    # create entries in db for item_types
    item_types = df["Item_Type"].unique().tolist()
    await prisma.item_types.delete_many()  # delete all existing entries first
    item_types_ids = [
        {"id": str(uuid.uuid4()), "name": item_type} for item_type in item_types
    ]
    await prisma.item_types.create_many(item_types_ids)  # type: ignore

    # replace item_type with id
    df["Item_Type"] = df["Item_Type"].replace(
        to_replace=item_types, value=[item["id"] for item in item_types_ids]
    )

    # rename columns
    df = df.rename(
        columns={
            "Item_Identifier": "id",
            "Item_Fat_Content": "fat_content_type",
            "Item_Type": "item_type",
            "Outlet_Identifier": "outlet_id",
        }
    )

    # create entries in db for items
    items = df[
        [
            "id",
            "fat_content_type",
            "item_type",
            "outlet_id",
        ]
    ].drop_duplicates("id")
    items = items.to_dict("records")
    await prisma.items.delete_many()
    await prisma.items.create_many(items)  # type: ignore


async def create_sales_in_db(df: pd.DataFrame):
    # rename columns
    df = df.rename(
        columns={
            "Item_Identifier": "item_id",
            "Item_Weight": "weight",
            "Item_Visibility": "visibility",
            "Item_MRP": "mrp",
            "Outlet_Identifier": "outlet_id",
            "Item_Outlet_Sales": "sales",
        }
    )

    # create entries in db for sales
    sales = df[
        [
            "item_id",
            "weight",
            "visibility",
            "mrp",
            "outlet_id",
            "sales",
        ]
    ].drop_duplicates()
    sales = sales.to_dict("records")
    await prisma.sales.delete_many()
    await prisma.sales.create_many(sales)  # type: ignore


@app.post("/process-cleaned-data")
async def process_cleaned_data():
    # read file from local
    df = pd.read_csv("upload/data.csv")
    del df[df.columns[0]]
    df = df.sort_values(by=["Item_Identifier", "Outlet_Identifier"]).reset_index(
        level=0, drop=True
    )

    # Standardize Item_Fat_Content column using jaro winkler distance
    texts_before_standardizing = df["Item_Fat_Content"].unique().tolist()
    normalized_texts = normalize_text(texts_before_standardizing, threshold=0.5)
    df["Item_Fat_Content"] = df["Item_Fat_Content"].replace(
        to_replace=texts_before_standardizing, value=normalized_texts
    )

    # Fill in missing data for Item_Weight column using interpolation
    # df["Item_Weight"].interpolate(method="polynomial", order=2, inplace=True)
    df["Item_Weight"] = (
        df.groupby(["Item_Identifier", "Item_Fat_Content", "Item_Type"])["Item_Weight"]
        .apply(lambda x: x.ffill().bfill().fillna(df["Item_Weight"].median()))
        .reset_index(drop=True)
    )

    outlet_sizes_df = (
        df[
            [
                "Outlet_Identifier",
                "Outlet_Size",
            ]
        ]
        .drop_duplicates()
        .fillna("Unknown")
    )

    # replace df outlet_size column with outlet_size from outlet_sizes_df
    df["Outlet_Size"] = df["Outlet_Identifier"].map(
        outlet_sizes_df.set_index("Outlet_Identifier")["Outlet_Size"]
    )

    # create outlets
    await create_outlets_in_db(df)

    # create items
    await create_items_in_db(df)

    # create sales
    await create_sales_in_db(df)

    return IBaseResponse(
        statusCode=HTTPStatus.OK,
        message="Cleaned data and added to DB successfully",
    )


@app.post("/item-stats")
async def item_stats():
    # read dirty data from local
    dirty_data = pd.read_csv("upload/data.csv")

    # read clean data from db
    items = pd.DataFrame(jsonable_encoder(await prisma.items.find_many()))
    fat_content_types = pd.DataFrame(
        jsonable_encoder(await prisma.fat_content_types.find_many())
    )

    # replace fat_content_type_id with name
    items = items.replace(
        to_replace=[item["id"] for item in fat_content_types.to_dict("records")],
        value=fat_content_types["name"].unique().tolist(),
    )

    # distinct items count
    distinct_items_count = await prisma.items.count()

    # item fat content text normalization
    texts_before_standardizing = dirty_data["Item_Fat_Content"].unique().tolist()
    normalized_texts = fat_content_types["name"].unique().tolist()

    # item fat content values count
    original_item_fat_content_types_count = (
        dirty_data.groupby("Item_Fat_Content")["Item_Identifier"].nunique().to_dict(),
    )[0]
    item_fat_content_types_count = (
        items.groupby("fat_content_type")["id"].nunique().to_dict(),
    )[0]

    # item types count
    item_types_count = await prisma.item_types.count()

    # payload
    payload = {
        "distinct_items_count": distinct_items_count,
        "item_types_count": item_types_count,
        "dirty_texts": texts_before_standardizing,
        "cleaned_texts": normalized_texts,
        "dirty_item_fat_content_types_count": original_item_fat_content_types_count,
        "cleaned_item_fat_content_types_count": item_fat_content_types_count,
    }

    return IBaseResponse(
        statusCode=HTTPStatus.OK,
        message="Item stats retrieved successfully",
        data=payload,
    )


@app.post("/outlet-stats")
async def outlet_stats():
    # read clean data from db
    outlets = pd.DataFrame(jsonable_encoder(await prisma.outlets.find_many()))
    distinct_outlet_counts = await prisma.outlets.count()
    outlet_size_types = pd.DataFrame(
        jsonable_encoder(await prisma.outlet_size_types.find_many())
    )
    outlet_location_types = pd.DataFrame(
        jsonable_encoder(await prisma.outlet_location_types.find_many())
    )
    outlet_types = pd.DataFrame(jsonable_encoder(await prisma.outlet_types.find_many()))

    # replace outlet_size_type_id with name
    outlets = outlets.replace(
        to_replace=[item["id"] for item in outlet_size_types.to_dict("records")],
        value=outlet_size_types["name"].unique().tolist(),
    )

    # replace outlet_location_type_id with name
    outlets = outlets.replace(
        to_replace=[item["id"] for item in outlet_location_types.to_dict("records")],
        value=outlet_location_types["name"].unique().tolist(),
    )

    # replace outlet_type_id with name
    outlets = outlets.replace(
        to_replace=[item["id"] for item in outlet_types.to_dict("records")],
        value=outlet_types["name"].unique().tolist(),
    )

    # outlet size value counts
    outlet_size_value_counts = (
        outlets.groupby("outlet_size_type")["id"].nunique().to_dict(),
    )[0]

    # payload
    payload = {
        "distinct_outlet_counts": distinct_outlet_counts,
        "outlet_size_value_counts": outlet_size_value_counts,
    }

    return IBaseResponse(
        statusCode=HTTPStatus.OK,
        message="Outlet stats retrieved successfully",
        data=payload,
    )


@app.post("/sales-stats")
async def sales_stats():
    # read clean data from db
    sales = pd.DataFrame(jsonable_encoder(await prisma.sales.find_many()))

    # cast column values
    sales["weight"] = sales["weight"].astype(float)
    sales["visibility"] = sales["visibility"].astype(float)
    sales["mrp"] = sales["mrp"].astype(float)
    sales["sales"] = sales["sales"].astype(float)

    # median_item_weight by item_id
    median_item_weight_df = sales[
        [
            "item_id",
            "weight",
        ]
    ].drop_duplicates()
    median_item_weight = median_item_weight_df["weight"].median()

    # item visibility
    median_item_visibility_by_outlet = (
        sales.groupby(["outlet_id"])["visibility"].median().to_dict()
    )

    # item MRP
    median_item_mrp_df = sales[
        [
            "item_id",
            "outlet_id",
            "mrp",
        ]
    ].drop_duplicates()
    median_item_mrp = median_item_mrp_df["mrp"].median()

    # median sales by outlet_id
    median_sales_by_outlet = sales.groupby(["outlet_id"])["sales"].median().to_dict()

    # payload
    payload = {
        "median_item_weight": median_item_weight,
        "median_item_visibility_by_outlet": median_item_visibility_by_outlet,
        "median_item_mrp": median_item_mrp,
        "median_sales_by_outlet": median_sales_by_outlet,
    }

    return IBaseResponse(
        statusCode=HTTPStatus.OK,
        message="Sales stats retrieved successfully",
        data=payload,
    )
