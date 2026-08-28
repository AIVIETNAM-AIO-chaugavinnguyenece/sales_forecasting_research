"""Generate synthetic Australian retail sales, weather and holiday data."""

import os
from datetime import date, timedelta

import numpy as np
import pandas as pd

# Set random seed for reproducibility.
# Note: the seed is set once here and the same Generator is threaded through
# every function. Calling np.random.seed() again inside a loop (or passing
# None) would silently reseed from OS entropy and make the output
# irreproducible.
RNG = np.random.default_rng(2025)

START_DATE = "2023-01-01"
END_DATE = "2025-12-31"
OUTPUT_DIR = "sales_forecasting_research/data"

# Australian cities and their states. One city per state keeps the join
# between sales (province) and weather (city) one-to-one.
CITY_TO_PROVINCE = {
    "Sydney": "New South Wales",
    "Melbourne": "Victoria",
    "Brisbane": "Queensland",
    "Adelaide": "South Australia",
}

PROVINCE_TO_CITY = {v: k for k, v in CITY_TO_PROVINCE.items()}

# Short codes used by the public holiday rules.
PROVINCE_CODE = {
    "New South Wales": "NSW",
    "Victoria": "VIC",
    "Queensland": "QLD",
    "South Australia": "SA",
}


def generate_store_data():
    """Generate store data"""
    stores = [
        # Sydney
        {
            "store_id": 1,
            "store_name": "Sydney CBD",
            "province": "New South Wales",
            "city": "Sydney",
            "size_factor": 1.30,
        },
        {
            "store_id": 2,
            "store_name": "Parramatta",
            "province": "New South Wales",
            "city": "Sydney",
            "size_factor": 1.05,
        },
        {
            "store_id": 3,
            "store_name": "Chatswood",
            "province": "New South Wales",
            "city": "Sydney",
            "size_factor": 0.90,
        },
        # Melbourne
        {
            "store_id": 4,
            "store_name": "Melbourne CBD",
            "province": "Victoria",
            "city": "Melbourne",
            "size_factor": 1.25,
        },
        {
            "store_id": 5,
            "store_name": "Footscray",
            "province": "Victoria",
            "city": "Melbourne",
            "size_factor": 1.00,
        },
        {
            "store_id": 6,
            "store_name": "Dandenong",
            "province": "Victoria",
            "city": "Melbourne",
            "size_factor": 0.85,
        },
        # Brisbane
        {
            "store_id": 7,
            "store_name": "Brisbane CBD",
            "province": "Queensland",
            "city": "Brisbane",
            "size_factor": 1.15,
        },
        {
            "store_id": 8,
            "store_name": "Chermside",
            "province": "Queensland",
            "city": "Brisbane",
            "size_factor": 0.95,
        },
        {
            "store_id": 9,
            "store_name": "Logan",
            "province": "Queensland",
            "city": "Brisbane",
            "size_factor": 0.80,
        },
        # Adelaide
        {
            "store_id": 10,
            "store_name": "Adelaide CBD",
            "province": "South Australia",
            "city": "Adelaide",
            "size_factor": 1.10,
        },
        {
            "store_id": 11,
            "store_name": "Modbury",
            "province": "South Australia",
            "city": "Adelaide",
            "size_factor": 0.90,
        },
        {
            "store_id": 12,
            "store_name": "Noarlunga",
            "province": "South Australia",
            "city": "Adelaide",
            "size_factor": 0.75,
        },
    ]
    return pd.DataFrame(stores)


def generate_item_data():
    """Generate item data"""
    # temp_effect: change in demand per degree above the seasonal norm.
    #   positive = sells more in hot weather, negative = sells more in cold.
    # peak_month: month of peak seasonal demand (Southern Hemisphere,
    #   so 1 = January = midsummer, 7 = July = midwinter).
    items = [
        # Staples
        {
            "item_id": 1,
            "item_name": "Long Grain Rice 2kg",
            "category": "Staples",
            "base_sales": 14,
            "base_price": 5.50,
            "elasticity": -0.55,
            "promo_rate": 0.10,
            "volatility": 0.20,
            "temp_effect": 0.000,
            "rain_effect": 0.02,
            "peak_month": 7,
        },
        {
            "item_id": 2,
            "item_name": "Dried Pasta 500g",
            "category": "Staples",
            "base_sales": 18,
            "base_price": 2.20,
            "elasticity": -0.70,
            "promo_rate": 0.14,
            "volatility": 0.22,
            "temp_effect": -0.004,
            "rain_effect": 0.05,
            "peak_month": 7,
        },
        {
            "item_id": 3,
            "item_name": "White Bread Loaf",
            "category": "Staples",
            "base_sales": 42,
            "base_price": 3.80,
            "elasticity": -0.45,
            "promo_rate": 0.08,
            "volatility": 0.18,
            "temp_effect": 0.000,
            "rain_effect": 0.03,
            "peak_month": 7,
        },
        {
            "item_id": 4,
            "item_name": "Plain Flour 1kg",
            "category": "Staples",
            "base_sales": 7,
            "base_price": 2.50,
            "elasticity": -0.60,
            "promo_rate": 0.10,
            "volatility": 0.28,
            "temp_effect": -0.008,
            "rain_effect": 0.08,
            "peak_month": 7,
        },
        {
            "item_id": 5,
            "item_name": "Breakfast Cereal 1kg",
            "category": "Staples",
            "base_sales": 15,
            "base_price": 5.20,
            "elasticity": -0.80,
            "promo_rate": 0.14,
            "volatility": 0.22,
            "temp_effect": -0.006,
            "rain_effect": 0.03,
            "peak_month": 7,
        },
        {
            "item_id": 6,
            "item_name": "White Sugar 1kg",
            "category": "Staples",
            "base_sales": 8,
            "base_price": 2.40,
            "elasticity": -0.50,
            "promo_rate": 0.08,
            "volatility": 0.25,
            "temp_effect": 0.000,
            "rain_effect": 0.02,
            "peak_month": 7,
        },
        # Dairy & Frozen
        {
            "item_id": 7,
            "item_name": "Full Cream Milk 2L",
            "category": "Dairy & Frozen",
            "base_sales": 55,
            "base_price": 3.30,
            "elasticity": -0.40,
            "promo_rate": 0.06,
            "volatility": 0.15,
            "temp_effect": 0.002,
            "rain_effect": 0.02,
            "peak_month": 1,
        },
        {
            "item_id": 8,
            "item_name": "Tasty Cheese Block 500g",
            "category": "Dairy & Frozen",
            "base_sales": 16,
            "base_price": 7.50,
            "elasticity": -0.95,
            "promo_rate": 0.18,
            "volatility": 0.22,
            "temp_effect": 0.000,
            "rain_effect": 0.03,
            "peak_month": 7,
        },
        {
            "item_id": 9,
            "item_name": "Greek Yoghurt 1kg",
            "category": "Dairy & Frozen",
            "base_sales": 20,
            "base_price": 5.50,
            "elasticity": -0.90,
            "promo_rate": 0.16,
            "volatility": 0.22,
            "temp_effect": 0.010,
            "rain_effect": 0.00,
            "peak_month": 1,
        },
        {
            "item_id": 10,
            "item_name": "Vanilla Ice Cream 2L",
            "category": "Dairy & Frozen",
            "base_sales": 14,
            "base_price": 7.00,
            "elasticity": -1.35,
            "promo_rate": 0.20,
            "volatility": 0.35,
            "temp_effect": 0.045,
            "rain_effect": -0.12,
            "peak_month": 1,
        },
        {
            "item_id": 11,
            "item_name": "Frozen Mixed Vegetables 1kg",
            "category": "Dairy & Frozen",
            "base_sales": 11,
            "base_price": 4.50,
            "elasticity": -0.75,
            "promo_rate": 0.14,
            "volatility": 0.25,
            "temp_effect": -0.006,
            "rain_effect": 0.09,
            "peak_month": 7,
        },
        {
            "item_id": 12,
            "item_name": "Frozen Meat Pies 4pk",
            "category": "Dairy & Frozen",
            "base_sales": 12,
            "base_price": 6.80,
            "elasticity": -1.00,
            "promo_rate": 0.16,
            "volatility": 0.28,
            "temp_effect": -0.025,
            "rain_effect": 0.14,
            "peak_month": 7,
        },
        # Beverages
        {
            "item_id": 13,
            "item_name": "Cola Soft Drink 1.25L",
            "category": "Beverages",
            "base_sales": 46,
            "base_price": 3.20,
            "elasticity": -1.45,
            "promo_rate": 0.24,
            "volatility": 0.25,
            "temp_effect": 0.035,
            "rain_effect": -0.06,
            "peak_month": 1,
        },
        {
            "item_id": 14,
            "item_name": "Orange Juice 2L",
            "category": "Beverages",
            "base_sales": 24,
            "base_price": 4.50,
            "elasticity": -1.15,
            "promo_rate": 0.18,
            "volatility": 0.24,
            "temp_effect": 0.020,
            "rain_effect": -0.02,
            "peak_month": 1,
        },
        {
            "item_id": 15,
            "item_name": "Spring Water 6L",
            "category": "Beverages",
            "base_sales": 32,
            "base_price": 4.00,
            "elasticity": -1.10,
            "promo_rate": 0.20,
            "volatility": 0.26,
            "temp_effect": 0.042,
            "rain_effect": -0.10,
            "peak_month": 1,
        },
        {
            "item_id": 16,
            "item_name": "Ground Coffee 1kg",
            "category": "Beverages",
            "base_sales": 9,
            "base_price": 22.00,
            "elasticity": -1.20,
            "promo_rate": 0.20,
            "volatility": 0.28,
            "temp_effect": -0.024,
            "rain_effect": 0.10,
            "peak_month": 7,
        },
        {
            "item_id": 17,
            "item_name": "Black Tea Bags 100pk",
            "category": "Beverages",
            "base_sales": 10,
            "base_price": 4.50,
            "elasticity": -0.85,
            "promo_rate": 0.14,
            "volatility": 0.26,
            "temp_effect": -0.020,
            "rain_effect": 0.11,
            "peak_month": 7,
        },
        {
            "item_id": 18,
            "item_name": "Sports Drink 600ml",
            "category": "Beverages",
            "base_sales": 9,
            "base_price": 4.20,
            "elasticity": -1.30,
            "promo_rate": 0.22,
            "volatility": 0.34,
            "temp_effect": 0.050,
            "rain_effect": -0.14,
            "peak_month": 1,
        },
        # Snacks
        {
            "item_id": 19,
            "item_name": "Potato Chips 175g",
            "category": "Snacks",
            "base_sales": 38,
            "base_price": 4.20,
            "elasticity": -1.50,
            "promo_rate": 0.26,
            "volatility": 0.26,
            "temp_effect": 0.015,
            "rain_effect": 0.07,
            "peak_month": 1,
        },
        {
            "item_id": 20,
            "item_name": "Sweet Biscuits 250g",
            "category": "Snacks",
            "base_sales": 28,
            "base_price": 3.50,
            "elasticity": -1.25,
            "promo_rate": 0.22,
            "volatility": 0.24,
            "temp_effect": -0.005,
            "rain_effect": 0.09,
            "peak_month": 7,
        },
        {
            "item_id": 21,
            "item_name": "Chocolate Block 180g",
            "category": "Snacks",
            "base_sales": 26,
            "base_price": 5.00,
            "elasticity": -1.40,
            "promo_rate": 0.28,
            "volatility": 0.28,
            "temp_effect": -0.018,
            "rain_effect": 0.08,
            "peak_month": 7,
        },
        {
            "item_id": 22,
            "item_name": "Mixed Nuts 400g",
            "category": "Snacks",
            "base_sales": 12,
            "base_price": 9.00,
            "elasticity": -1.15,
            "promo_rate": 0.20,
            "volatility": 0.26,
            "temp_effect": 0.000,
            "rain_effect": 0.05,
            "peak_month": 7,
        },
        {
            "item_id": 23,
            "item_name": "Muesli Bars 6pk",
            "category": "Snacks",
            "base_sales": 17,
            "base_price": 4.80,
            "elasticity": -1.10,
            "promo_rate": 0.20,
            "volatility": 0.24,
            "temp_effect": 0.004,
            "rain_effect": 0.04,
            "peak_month": 1,
        },
        # Household
        {
            "item_id": 24,
            "item_name": "Toilet Paper 12pk",
            "category": "Household",
            "base_sales": 15,
            "base_price": 9.00,
            "elasticity": -0.90,
            "promo_rate": 0.18,
            "volatility": 0.22,
            "temp_effect": 0.000,
            "rain_effect": 0.04,
            "peak_month": 7,
        },
        {
            "item_id": 25,
            "item_name": "Laundry Liquid 2L",
            "category": "Household",
            "base_sales": 9,
            "base_price": 12.00,
            "elasticity": -1.05,
            "promo_rate": 0.22,
            "volatility": 0.26,
            "temp_effect": 0.000,
            "rain_effect": 0.04,
            "peak_month": 7,
        },
        {
            "item_id": 26,
            "item_name": "Dishwashing Liquid 500ml",
            "category": "Household",
            "base_sales": 10,
            "base_price": 4.50,
            "elasticity": -0.75,
            "promo_rate": 0.14,
            "volatility": 0.25,
            "temp_effect": 0.000,
            "rain_effect": 0.03,
            "peak_month": 7,
        },
        {
            "item_id": 27,
            "item_name": "Paper Towel 4pk",
            "category": "Household",
            "base_sales": 11,
            "base_price": 5.50,
            "elasticity": -0.80,
            "promo_rate": 0.16,
            "volatility": 0.25,
            "temp_effect": 0.000,
            "rain_effect": 0.05,
            "peak_month": 7,
        },
        {
            "item_id": 28,
            "item_name": "Sunscreen SPF50+ 200ml",
            "category": "Household",
            "base_sales": 7,
            "base_price": 11.00,
            "elasticity": -1.10,
            "promo_rate": 0.18,
            "volatility": 0.40,
            "temp_effect": 0.060,
            "rain_effect": -0.16,
            "peak_month": 1,
        },
        {
            "item_id": 29,
            "item_name": "Insect Repellent 150ml",
            "category": "Household",
            "base_sales": 5,
            "base_price": 8.50,
            "elasticity": -0.95,
            "promo_rate": 0.14,
            "volatility": 0.40,
            "temp_effect": 0.045,
            "rain_effect": 0.10,
            "peak_month": 1,
        },
        {
            "item_id": 30,
            "item_name": "Firelighters 24pk",
            "category": "Household",
            "base_sales": 4,
            "base_price": 5.50,
            "elasticity": -0.70,
            "promo_rate": 0.10,
            "volatility": 0.45,
            "temp_effect": -0.045,
            "rain_effect": 0.20,
            "peak_month": 7,
        },
    ]
    return pd.DataFrame(items)


# Mean daily temperature by city and month, January to December.
# Southern Hemisphere: warmest in January, coldest in July.
CITY_TEMPERATURE = {
    "Sydney": [23.0, 23.0, 21.7, 19.2, 16.1, 13.7, 12.8, 14.0, 16.5, 18.5, 20.3, 22.1],
    "Melbourne": [
        20.1,
        20.2,
        18.6,
        15.6,
        12.7,
        10.5,
        9.8,
        10.9,
        12.7,
        14.6,
        16.6,
        18.6,
    ],
    "Brisbane": [
        25.8,
        25.5,
        24.4,
        22.1,
        18.9,
        16.4,
        15.5,
        16.5,
        19.4,
        21.7,
        23.6,
        25.0,
    ],
    "Adelaide": [
        23.3,
        23.4,
        20.9,
        17.7,
        14.8,
        12.2,
        11.4,
        12.5,
        14.5,
        16.8,
        19.7,
        21.4,
    ],
}

# Mean relative humidity by city and month.
CITY_HUMIDITY = {
    "Sydney": [70, 72, 72, 70, 71, 71, 68, 63, 63, 64, 67, 68],
    "Melbourne": [59, 62, 63, 67, 74, 78, 76, 71, 66, 62, 60, 59],
    "Brisbane": [70, 72, 71, 69, 68, 67, 63, 59, 60, 63, 65, 67],
    "Adelaide": [46, 47, 50, 57, 67, 72, 72, 67, 60, 54, 49, 46],
}

# Day-to-day temperature variability. Melbourne is famously the most volatile.
CITY_VOLATILITY = {"Sydney": 3.0, "Melbourne": 4.6, "Brisbane": 2.5, "Adelaide": 4.2}


def get_season(month):
    """Get the Australian season for a month"""
    if month in (12, 1, 2):
        return "Summer"
    if month in (3, 4, 5):
        return "Autumn"
    if month in (6, 7, 8):
        return "Winter"
    return "Spring"


def generate_weather_data(dates):
    """Generate weather data"""
    rows = []
    for city in CITY_TO_PROVINCE:
        # Persistent day-to-day anomaly. Independent daily draws would make
        # temperature behave like white noise, which is neither realistic
        # nor useful as a predictor.
        anomaly = 0.0
        for current_date in dates:
            month = current_date.month
            norm = CITY_TEMPERATURE[city][month - 1]
            volatility = CITY_VOLATILITY[city]

            anomaly = 0.7 * anomaly + RNG.normal(0, volatility * 0.71)
            temperature = norm + anomaly

            humidity = CITY_HUMIDITY[city][month - 1] + RNG.normal(0, 6)
            humidity = float(np.clip(humidity, 25, 98))

            rows.append(
                {
                    "date": current_date,
                    "city": city,
                    "province": CITY_TO_PROVINCE[city],
                    "temperature": round(temperature, 1),
                    "humidity": round(humidity, 1),
                    "season": get_season(month),
                }
            )
    return pd.DataFrame(rows)


def easter_sunday(year):
    """Calculate Easter Sunday using the anonymous Gregorian algorithm"""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    m = (32 + 2 * e + 2 * i - h - k) % 7
    n = (a + 11 * h + 22 * m) // 451
    month, day = divmod(h + m - 7 * n + 114, 31)
    return date(year, month, day + 1)


def nth_weekday(year, month, weekday, n):
    """Get the n-th weekday of a month (Monday = 0)"""
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def get_public_holidays(year):
    """Get Australian public holidays as (date, name, provinces)"""
    easter = easter_sunday(year)
    every_state = ("NSW", "VIC", "QLD", "SA")

    holidays = [
        (date(year, 1, 1), "New Year's Day", every_state),
        (date(year, 1, 26), "Australia Day", every_state),
        (easter - timedelta(days=2), "Good Friday", every_state),
        (easter - timedelta(days=1), "Easter Saturday", every_state),
        (easter + timedelta(days=1), "Easter Monday", every_state),
        (date(year, 4, 25), "ANZAC Day", every_state),
        (date(year, 12, 25), "Christmas Day", every_state),
        (date(year, 12, 26), "Boxing Day", every_state),
        # Labour Day falls in different months in different states.
        (nth_weekday(year, 3, 0, 2), "Labour Day", ("VIC",)),
        (nth_weekday(year, 5, 0, 1), "Labour Day", ("QLD",)),
        (nth_weekday(year, 10, 0, 1), "Labour Day", ("NSW", "SA")),
        # King's Birthday is June everywhere except Queensland.
        (nth_weekday(year, 6, 0, 2), "King's Birthday", ("NSW", "VIC", "SA")),
        (nth_weekday(year, 10, 0, 1), "King's Birthday", ("QLD",)),
        (nth_weekday(year, 3, 0, 2), "Adelaide Cup", ("SA",)),
        (nth_weekday(year, 11, 1, 1), "Melbourne Cup", ("VIC",)),
    ]
    return holidays


def get_school_holidays(year):
    """Get approximate school holiday windows as (start, end) pairs"""
    easter = easter_sunday(year)
    return [
        (date(year, 1, 1), date(year, 1, 28)),
        (easter - timedelta(days=7), easter + timedelta(days=7)),
        (date(year, 6, 28), date(year, 7, 14)),
        (date(year, 9, 21), date(year, 10, 6)),
        (date(year, 12, 18), date(year, 12, 31)),
    ]


def generate_holiday_data(dates):
    """Generate holiday and calendar data"""
    years = range(dates[0].year, dates[-1].year + 1)

    holiday_lookup = {}
    for year in years:
        for holiday_date, name, provinces in get_public_holidays(year):
            for code in provinces:
                holiday_lookup.setdefault((holiday_date, code), name)

    school_days = set()
    for year in years:
        for start, end in get_school_holidays(year):
            current = start
            while current <= end:
                school_days.add(current)
                current += timedelta(days=1)

    rows = []
    for current_date in dates:
        plain_date = current_date.date()
        for province, code in PROVINCE_CODE.items():
            name = holiday_lookup.get((plain_date, code))
            rows.append(
                {
                    "date": current_date,
                    "province": province,
                    "day_of_week": current_date.day_name(),
                    "is_weekend": int(current_date.dayofweek >= 5),
                    "is_public_holiday": int(name is not None),
                    "holiday_name": name if name else "",
                    "is_school_holiday": int(plain_date in school_days),
                }
            )
    return pd.DataFrame(rows)


# How much each holiday changes sales. Christmas Day is near-total closure;
# Boxing Day is the biggest trading day of the Australian retail year.
HOLIDAY_EFFECT = {
    "New Year's Day": 0.70,
    "Australia Day": 1.20,
    "Good Friday": 0.55,
    "Easter Saturday": 1.25,
    "Easter Monday": 0.85,
    "ANZAC Day": 0.75,
    "King's Birthday": 0.95,
    "Labour Day": 0.95,
    "Adelaide Cup": 0.95,
    "Melbourne Cup": 1.05,
    "Christmas Day": 0.10,
    "Boxing Day": 1.40,
}

# Sales by day of week, Monday to Sunday. Australian grocery peaks Thursday
# to Saturday.
WEEKDAY_EFFECT = [0.94, 0.92, 0.98, 1.10, 1.24, 1.30, 1.04]


# Promotion discount depths and how often each is used.
PROMO_DEPTHS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.50]
PROMO_WEIGHTS = [0.22, 0.24, 0.22, 0.16, 0.12, 0.04]

# Promotion type is assigned from the discount depth, and each type carries
# its own lift on top of the pure price response. A catalogue feature reaches
# more shoppers than a shelf ticket, so the type column carries real
# information rather than being a decorative label.
PROMO_TYPES = {
    "In-Store Display": 1.10,
    "Catalogue": 1.30,
    "Clearance": 1.15,
}


def classify_promo(depth):
    """Assign a promotion type from its discount depth"""
    if depth >= 0.40:
        return "Clearance"
    if depth >= 0.20:
        return "Catalogue"
    return "In-Store Display"


# Yearly shelf price inflation.
PRICE_INFLATION = 0.045


def generate_promo_schedule(n_days, promo_rate, dates, store, item, next_id):
    """Generate a promotion schedule as contiguous runs, plus campaign records"""
    # Promotions run for a week or a fortnight at a time. Flipping an
    # independent coin each day would make the promotion flag flicker on and
    # off in a way no retailer ever operates, and would make price trivially
    # easy for a model to decode.
    discount = np.zeros(n_days)
    promo_id = np.zeros(n_days, dtype=int)
    campaigns = []
    target_days = int(promo_rate * n_days)
    placed = 0
    attempts = 0

    while placed < target_days and attempts < 400:
        attempts += 1
        run_length = int(RNG.integers(5, 15))
        start = int(RNG.integers(0, max(1, n_days - run_length)))

        # Leave a gap between runs on the same item.
        if discount[max(0, start - 14) : start + run_length + 14].any():
            continue

        depth = float(RNG.choice(PROMO_DEPTHS, p=PROMO_WEIGHTS))
        discount[start : start + run_length] = depth
        promo_id[start : start + run_length] = next_id
        end = min(start + run_length - 1, n_days - 1)

        campaigns.append(
            {
                "promo_id": next_id,
                "store_id": store["store_id"],
                "store_name": store["store_name"],
                "province": store["province"],
                "item_id": item["item_id"],
                "item_name": item["item_name"],
                "category": item["category"],
                "start_date": dates[start],
                "end_date": dates[end],
                "duration_days": end - start + 1,
                "discount_pct": round(depth * 100, 1),
                "promo_type": classify_promo(depth),
            }
        )
        next_id += 1
        placed += run_length

    return discount, promo_id, campaigns, next_id


def calculate_daily_sales(
    item, store, weather, calendar, current_date, discount, promo_type
):
    """Calculate the sales for one item, in one store, on one day"""
    month = current_date.month

    # Baseline scaled by store size.
    sales = item["base_sales"] * store["size_factor"]

    # Seasonal cycle, peaking in the item's peak month.
    month_offset = (month - item["peak_month"]) * (2 * np.pi / 12)
    sales *= 1 + 0.30 * np.cos(month_offset)

    # Day of week.
    sales *= WEEKDAY_EFFECT[current_date.dayofweek]

    # Weather. Temperature is measured against the local seasonal norm, so a
    # 25 degree day counts as hot in Melbourne and mild in Brisbane.
    norm = CITY_TEMPERATURE[store["city"]][month - 1]
    sales *= 1 + item["temp_effect"] * (weather["temperature"] - norm)
    if weather["humidity"] > 80:
        sales *= 1 + item["rain_effect"]

    # Public and school holidays.
    if calendar["is_public_holiday"]:
        sales *= HOLIDAY_EFFECT.get(calendar["holiday_name"], 1.0)
    if calendar["is_school_holiday"]:
        sales *= 1.07

    # Christmas build-up over the three weeks before 25 December.
    if month == 12 and 4 <= current_date.day <= 24:
        sales *= 1 + 0.45 * (current_date.day - 4) / 20

    # Price. A discount raises demand according to the item's elasticity, so
    # staples barely move while snacks and soft drinks respond strongly.
    if discount > 0:
        sales *= (1 - discount) ** item["elasticity"]
        sales *= PROMO_TYPES[promo_type]

    # Slow annual growth.
    sales *= 1.03 ** (current_date.year - 2023)

    # Random variation. A Poisson draw keeps the result a non-negative
    # integer without the artificial pile-up at zero that clipping a normal
    # distribution would produce.
    sales *= RNG.normal(1.0, item["volatility"] * 0.5)
    return int(RNG.poisson(max(sales, 0.05)))


def generate_sales_data(stores, items, weather, calendar, dates):
    """Generate sales data"""
    weather_lookup = weather.set_index(["date", "city"]).to_dict("index")
    calendar_lookup = calendar.set_index(["date", "province"]).to_dict("index")

    n_days = len(dates)
    rows = []
    campaigns = []
    next_id = 1

    for _, store in stores.iterrows():
        print(f"  generating store {store['store_id']}: {store['store_name']}")
        for _, item in items.iterrows():
            # One promotion schedule per store and item, decided once rather
            # than re-rolled every day.
            discounts, promo_ids, item_campaigns, next_id = generate_promo_schedule(
                n_days, item["promo_rate"], dates, store, item, next_id
            )
            campaigns.extend(item_campaigns)
            type_by_id = {c["promo_id"]: c["promo_type"] for c in item_campaigns}

            for day_index, current_date in enumerate(dates):
                today_weather = weather_lookup[(current_date, store["city"])]
                today_calendar = calendar_lookup[(current_date, store["province"])]
                discount = discounts[day_index]
                promo_id = int(promo_ids[day_index])
                promo_type = type_by_id.get(promo_id, "")

                years_elapsed = (current_date - dates[0]).days / 365.25
                base_price = item["base_price"] * (1 + PRICE_INFLATION) ** years_elapsed
                price = base_price * (1 - discount)

                rows.append(
                    {
                        "date": current_date,
                        "province": store["province"],
                        "store_id": store["store_id"],
                        "store_name": store["store_name"],
                        "category": item["category"],
                        "item_id": item["item_id"],
                        "item_name": item["item_name"],
                        "price": round(price, 2),
                        "base_price": round(base_price, 2),
                        "is_promotion": int(discount > 0),
                        "promo_id": promo_id,
                        "sales": calculate_daily_sales(
                            item,
                            store,
                            today_weather,
                            today_calendar,
                            current_date,
                            discount,
                            promo_type if promo_type else "In-Store Display",
                        ),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(campaigns)


def add_outliers_and_nans(df, outlier_rate=0.002, nan_rate=0.01):
    """Add outliers and missing values to the sales data"""
    df = df.copy()
    n_rows = len(df)

    n_outliers = int(n_rows * outlier_rate)
    outlier_rows = RNG.choice(n_rows, size=n_outliers, replace=False)
    df.iloc[outlier_rows, df.columns.get_loc("sales")] *= 3
    print(f"  added {n_outliers:,} outliers ({outlier_rate:.1%})")

    df["sales"] = df["sales"].astype("float64")
    n_nans = int(n_rows * nan_rate)
    nan_rows = RNG.choice(n_rows, size=n_nans, replace=False)
    df.iloc[nan_rows, df.columns.get_loc("sales")] = np.nan
    print(f"  added {n_nans:,} missing values ({nan_rate:.1%})")

    return df


def check_missing_values(df, name):
    """Check and report missing values in a dataframe"""
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) == 0:
        print(f"  {name}: no missing values")
    else:
        for column, count in missing.items():
            print(f"  {name}: {column} has {count:,} missing ({count / len(df):.2%})")


def main():
    """Generate all three datasets and write them to CSV"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    print(f"Generating data from {START_DATE} to {END_DATE} ({len(dates)} days)\n")

    stores = generate_store_data()
    items = generate_item_data()
    print(f"Stores: {len(stores)}  Items: {len(items)}\n")

    print("Generating weather data ...")
    weather = generate_weather_data(dates)

    print("Generating holiday data ...")
    calendar = generate_holiday_data(dates)

    print("Generating sales and promotion data ...")
    sales, promotions = generate_sales_data(stores, items, weather, calendar, dates)
    promotions = promotions.sort_values("promo_id").reset_index(drop=True)

    print("\nAdding data quality issues ...")
    sales = add_outliers_and_nans(sales)

    print("\nChecking missing values ...")
    check_missing_values(sales, "sales")
    check_missing_values(weather, "weather")
    check_missing_values(calendar, "holiday")
    check_missing_values(promotions, "promotion")

    print(f"\nWriting to {OUTPUT_DIR}/ ...")
    for name, df in [
        ("sales_data", sales),
        ("weather_data", weather),
        ("holiday_data", calendar),
        ("promotion_data", promotions),
    ]:
        path = os.path.join(OUTPUT_DIR, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"  {name + '.csv':20s} {len(df):>9,} rows  {df.shape[1]} cols")

    print("\nDone.")


if __name__ == "__main__":
    main()
