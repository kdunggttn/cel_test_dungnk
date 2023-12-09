# CEL - Test - Dung Nguyen Khac

To run the containers:

```
docker-compose build && docker-compose up -d
```

Go to http://localhost:8501 to see the Streamlit frontend.

Use the CSV file supplied with the Test on the Streamlit web app to see the results.

## Tech stacks:

### Backend

-   FastAPI as API server, with base response mapping (IBaseResponse)
-   Dedicated healthcheck endpoint for docker-compose (/healthcheck)
-   Prisma ORM with separate schema file (schema.prisma)
-   PostgreSQL as database

### Frontend

-   Streamlit as frontend

### Docker

-   Docker-compose to run the containers

## Details on the project:

-   All of the data processing is done in the backend. I created 2 tabs: Dirty and Cleaned. The dirty tab has metrics, charts... created using unprocessed data directly from the CSV file.
-   The cleaned tab is the data after processing. First, the backend receives the uploaded CSV file, then it will process the data and save it to the database. Then, the frontend will then fetch the data from the backend and display it.
-   I noticed that most of the missing data is random (meaning one row may have missing data, but other rows may not). Therefore, I filled in the missing data using other rows' values. Applicable to these columns: `Item_Weight`, `Outlet_Size`.
-   For the `Item_Fat_Content` column, I noticed that there are 2 values that are the same but written differently: `Low Fat` and `LF`, `Regular` and `reg`. I used Jaro-Winkler distance to normalise these values. The final results were shown on the web app.
-   After normalising these data, I added them to the DB. Subsequent requests are made to the DB (and not the CSV file) to fetch the data and display it on the frontend.
-   You can check the DB schema in the `backend/prisma/schema.prisma` file.

## Possible improvements:

-   Add exception filters to the backend
-   Better structure for the backend (separate the controllers, services, models...), as well as the frontend (separate the components, pages...)
-   Build the FE using ReactJS (I used Streamlit because of time contraints)
-   Explore more relationships in the data (e.g. Item visibility by Item type and Outlet; Item sales by Outlet type, Item type, Outlet location...)
