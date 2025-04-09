# Urban Heat Index Explorer

## Project Overview

An interactive climate dashboard built using Dash and Plotly to explore long-term trends in Heat Load Index (HLI) across the Philippines from 1950 to 2025. The application features choropleth maps, interactive charts, and guided onboarding to help users visualize regional heat patterns and environmental data over time.

This project is a final requirement submission for **DAT101M (Data Science Fundamentals)**.

You can explore the live version here: [https://dat101m-final-project.onrender.com/](https://dat101m-final-project.onrender.com/)

## Who Should Read This?

Ideal for:

- Students and educators in data science or environmental studies

- Researchers analyzing long-term temperature trends

- Developers working on interactive geospatial dashboards

## Installation Guide

1. Clone the repository:

```bash
git clone https://github.com/juliangoph/dat101m-final-project.git
cd dat101m-final-project
```

2. (Optional) Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run the App

Make sure the following files are present:

- `data/processed_philippine_cities_monthly.csv`
- `data/phl_adm_simple_maps.gpkg`
- Onboarding images in `/assets/`

Then launch:

```bash
python app.py
```

Visit [http://127.0.0.1:8050](http://127.0.0.1:8050) to explore the app.

## Folder Structure

```
├── app.py
├── requirements.txt
├── assets/                # Onboarding images and styles
├── data/
│   ├── processed_philippine_cities_monthly.csv
│   └── phl_adm_simple_maps.gpkg
├── README.md              # You're reading it!
```

## Key Features

- Dynamic map with decade slider and play/pause
- Region-specific HLI trend analysis
- Monthly HLI patterns across decades
- Dual-axis charts for temperature, wind, and radiation
- Guided onboarding with screenshots
