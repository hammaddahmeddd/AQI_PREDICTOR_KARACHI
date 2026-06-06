from datetime import datetime, timedelta

LATITUDE  = 24.8607
LONGITUDE = 67.0011

# Use yesterday as the END to avoid incomplete trailing hours
END_DATE   = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
# Go back to July 2023 for a ~2-year training window
START_DATE = "2023-07-04"