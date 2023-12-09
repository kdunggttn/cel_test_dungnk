import streamlit as st
import requests
import pandas as pd
import json

# streamlit run frontend/app.py --server.port 8500 --server.runOnSave true


def tabs(st):
    dirty, cleaned = st.tabs(["Dirty", "Cleaned"])

    with dirty:
        dirty_tab(st)

    with cleaned:
        cleaned_tab(st)


def process_dirty_data(st):
    st.subheader("Step 2: Data exploration")
    st.toast("Processing data...")

    req = requests.post("http://backend:8000/process-dirty-data")
    res = req.json()
    statusCode = res["statusCode"]
    message = res["message"]
    error = res["error"]
    data = json.loads(res["data"])

    if res["statusCode"] == 200:
        st.success(message)
        st.toast(message)

        (
            distinct_items_count,
            mean_item_weight,
            mean_item_visibility,
            mean_item_mrp,
        ) = st.columns(4)
        distinct_items_count.metric(
            "Distinct items", data["distinct_items_count"], "items", delta_color="off"
        )
        mean_item_weight.metric(
            "mean item weight",
            data["mean_item_weight"],
            "kg",
            delta_color="off",
        )
        mean_item_visibility.metric(
            "mean item visibility",
            data["mean_item_visibility"],
            "%",
            delta_color="off",
        )
        mean_item_mrp.metric(
            "mean item MRP",
            data["mean_item_mrp"],
            "$",
            delta_color="off",
        )

        st.caption("Chart 1. Item fat content types")
        st.bar_chart(data["item_fat_content_types"])

        (
            distinct_outlets_count,
            mean_item_outlet_sales_value,
            oldest_outlet_establishment_year,
            newest_outlet_establishment_year,
        ) = st.columns(4)
        distinct_outlets_count.metric(
            "Distinct outlets",
            data["distinct_outlets_count"],
            "outlets",
            delta_color="off",
        )
        mean_item_outlet_sales_value.metric(
            "mean item outlet sales value",
            round(float(data["mean_item_outlet_sales_value"]), 2),
            delta_color="off",
        )
        oldest_outlet_establishment_year.metric(
            "Oldest outlet establishment year",
            data["oldest_outlet_establishment_year"],
            delta_color="off",
        )
        newest_outlet_establishment_year.metric(
            "Newest outlet establishment year",
            data["newest_outlet_establishment_year"],
            delta_color="off",
        )

        st.caption("Chart 2. Outlet size types")
        st.bar_chart(data["outlet_size_types"])

        st.caption("Chart 3. Outlet location types")
        st.bar_chart(data["outlet_location_types"])

        st.caption("Chart 4. Outlet types")
        st.bar_chart(data["outlet_types"])

    else:
        st.error(message)
        st.toast(f"[{statusCode}] Error: {error} - {message}")


def process_cleaned_data(st):
    st.subheader("Step 2: Data exploration")
    st.toast("Processing data...")

    req = requests.post("http://backend:8000/process-cleaned-data")
    res = req.json()
    statusCode = res["statusCode"]
    message = res["message"]
    error = res["error"]

    if res["statusCode"] == 200:
        st.success(message)
        st.toast(message)

        st.subheader("Item stats")
        req = requests.post("http://backend:8000/item-stats")
        res = req.json()
        data = res["data"]

        st.metric(
            "Distinct items",
            data["distinct_items_count"],
            delta_color="off",
        )
        st.metric(
            "Types of items",
            data["item_types_count"],
            delta_color="off",
        )
        st.caption("Item fat contents before cleaning:")
        st.code(data["dirty_texts"])
        st.bar_chart(data["dirty_item_fat_content_types_count"])
        st.caption("Item fat contents after cleaning using Jaro-Wrinkler distance:")
        st.code(data["cleaned_texts"])
        st.bar_chart(data["cleaned_item_fat_content_types_count"])
        st.text(
            """
        We can see that most products (2/3) are Low Fat, and the rest are Regular.
        """
        )

        st.subheader("Outlet stats")
        req = requests.post("http://backend:8000/outlet-stats")
        res = req.json()
        data = res["data"]

        st.metric(
            "Distinct outlets",
            data["distinct_outlet_counts"],
            delta_color="off",
        )
        st.caption("Outlet sizes:")
        st.bar_chart(data["outlet_size_value_counts"])
        st.text(
            """
        While only 1 outlet is Large, most of them are Medium and Small.
        """
        )

        st.subheader("Sales stats")
        req = requests.post("http://backend:8000/sales-stats")
        res = req.json()
        data = res["data"]

        st.metric(
            "Median item weight",
            data["median_item_weight"],
            delta_color="off",
        )
        st.metric(
            "Median item MRP",
            data["median_item_mrp"],
            delta_color="off",
        )
        st.caption("Median item visibility by outlet:")
        st.bar_chart(data["median_item_visibility_by_outlet"])
        st.text(
            """
        It is evident that OUT010 and OUT019 have the highest median visibility
        across all items (nearly 9%), while the rest are somewhat similar (around 5%).
        From the data, these outlets are both Small (outlet size) Grocery
        Stores (outlet type) located in Tier 1 neighborhoods (outlet location type).
        These can explain why their median visibility is higher than the rest.
        """
        )
        st.caption("Median sales by outlet:")
        st.bar_chart(data["median_sales_by_outlet"])
        st.text(
            """
        In contrast, in terms of sales, OUT027 is the clear winner, with a median 
        sales of over 3300, while the rest are around 2000. From the data, this outlet 
        is a Medium Supermarket Type 3 located in Tier 3 neighborhoods. For OUT010 and 
        OUT019, despite their high item visibility, their median sales are the lowest, 
        which is understandable since they are both Small Grocery Stores.
        """
        )

    else:
        st.error(message)
        st.toast(f"[{statusCode}] Error: {error} - {message}")


def dirty_tab(st):
    st.header("Dirty data - Data exploration & mining")

    st.subheader("Step 1: Upload data")
    uploaded_file = st.file_uploader("Choose a CSV file", key="dirty")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df)

        # Send file to backend
        st.toast("Uploading file to backend...")
        req = requests.post("http://backend:8000/upload", json=df.to_json())
        res = req.json()
        statusCode = res["statusCode"]
        message = res["message"]
        error = res["error"]

        if statusCode == 200:
            st.success(message)
            st.toast(message)

            process_dirty_data(st)
        else:
            st.error(message)
            st.toast(f"[{statusCode}] Error: {error} - {message}")


def cleaned_tab(st):
    st.header("Cleaned data - Data exploration & mining")

    st.subheader("Step 1: Upload data")
    uploaded_file = st.file_uploader("Choose a CSV file", key="cleaned")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df)

        # Send file to backend
        st.toast("Uploading file to backend...")
        req = requests.post("http://backend:8000/upload", json=df.to_json())
        res = req.json()
        statusCode = res["statusCode"]
        message = res["message"]
        error = res["error"]

        if statusCode == 200:
            st.success(message)
            st.toast(message)

            process_cleaned_data(st)
        else:
            st.error(message)
            st.toast(f"[{statusCode}] Error: {error} - {message}")


def main():
    st.set_page_config(
        page_title="CEL - Test - Dung Nguyen Khac",
        page_icon="🧊",
    )
    st.title("CEL - Test - Dung Nguyen Khac")

    # Define tabs
    tabs(st)


if __name__ == "__main__":
    main()
