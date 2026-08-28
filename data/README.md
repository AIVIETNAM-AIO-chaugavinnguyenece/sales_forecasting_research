# Australian Retail Sales — Synthetic Dataset

Four CSV files. Sydney, Melbourne, Brisbane, Adelaide. 2023–2025.

```bash
python data_generator.py
```

| File | Rows | Columns |
|---|---|---|
| `sales_data.csv` | 394,560 | date, province, store_id, store_name, category, item_id, item_name, price, base_price, is_promotion, promo_id, sales |
| `weather_data.csv` | 4,384 | date, city, temperature, humidity, season |
| `holiday_data.csv` | 4,384 | date, province, day_of_week, is_weekend, is_public_holiday, holiday_name, is_school_holiday |
| `promotion_data.csv` | 7,177 | promo_id, store_id, store_name, province, item_id, item_name, category, start_date, end_date, duration_days, discount_pct, promo_type |

12 stores (3 per city), 30 items across 5 categories, 1,096 days (2024 is a leap year). 360 store-item series.

## Joining the tables

`province` and `city` are one-to-one:

| province | city |
|---|---|
| New South Wales | Sydney |
| Victoria | Melbourne |
| Queensland | Brisbane |
| South Australia | Adelaide |

```python
import pandas as pd

CITY = {
    "New South Wales": "Sydney",
    "Victoria": "Melbourne",
    "Queensland": "Brisbane",
    "South Australia": "Adelaide",
}

sales = pd.read_csv("sales_data.csv", parse_dates=["date"])
weather = pd.read_csv("weather_data.csv", parse_dates=["date"])
holiday = pd.read_csv("holiday_data.csv", parse_dates=["date"])

sales["city"] = sales["province"].map(CITY)
df = sales.merge(weather, on=["date", "city"], how="left")
df = df.merge(holiday, on=["date", "province"], how="left")
```

## The promotion table

`promotion_data.csv` holds one row per **campaign**, not per day. A daily promotion
file would be 394,560 rows — `sales_data` with the sales column removed — so it is
stored the way a real retail system stores it: as events with a start and end date.

Join it to sales through `promo_id`, which is 0 when nothing is running:

```python
promo = pd.read_csv("promotion_data.csv", parse_dates=["start_date", "end_date"])
df = sales.merge(promo, on="promo_id", how="left")
```

7,177 campaigns, 5 to 14 days each (mean 9.4). Type is assigned from discount depth,
and each type carries a different lift, so the column is informative rather than a
label:

| promo_type | Campaigns | Mean discount | Observed lift |
|---|---|---|---|
| In-Store Display | 3,311 | 12.5% | 1.26x |
| Catalogue | 3,582 | 24.0% | 1.75x |
| Clearance | 284 | 50.0% | 2.31x |

`is_promotion` in `sales_data` is exactly `promo_id > 0`; the redundancy is kept
for convenience so simple models need no join.

## Price and promotion columns

| Column | Meaning |
|---|---|
| `base_price` | Undiscounted shelf price in AUD, rising 4.5% a year |
| `price` | What the item actually sold for that day |
| `is_promotion` | 1 while a promotion is running |

Discount depth is `1 - price / base_price`, one of 10, 15, 20, 25, 30 or 50%.

Promotions run as contiguous blocks of 5 to 14 days with a gap between runs, not
as independent daily coin flips. Real retailers do not switch a promotion on and
off day by day, and daily flips would make the promotion flag trivially easy for
a model to decode.

Each item has its own price elasticity, so the promotional lift differs by
category rather than being a flat multiplier:

| Item | Off promo | On promo | Lift |
|---|---|---|---|
| Potato Chips 175g | 43.9 | 75.4 | 1.72x |
| Cola Soft Drink 1.25L | 53.1 | 88.1 | 1.66x |
| White Bread Loaf | 47.8 | 62.8 | 1.31x |
| Long Grain Rice 2kg | 15.9 | 21.8 | 1.37x |

Staples barely respond; snacks and soft drinks respond strongly. That contrast is
what makes price worth including as a feature.

## What drives sales

Multiplicative, in this order:

1. Item base level × store size factor
2. Seasonal cycle peaking in the item's `peak_month` (Southern Hemisphere)
3. Day of week — peaks Thursday to Saturday
4. Temperature, measured against the **local** seasonal norm, so 25 °C is hot
   in Melbourne and mild in Brisbane
5. Humidity above 80% as a wet-weather proxy
6. Public and school holidays
7. Christmas build-up over the three weeks before 25 December
8. **Price, via each item's own elasticity, plus a type-specific promotion lift**
9. 3% annual growth, with 2023 as the baseline year
10. Poisson noise

## The holiday file is the interesting one

Labour Day falls on a different date in every state:

| province | Labour Day 2023 |
|---|---|
| Victoria | 10 March |
| Queensland | 1 May |
| New South Wales | 2 October |
| South Australia | 2 October |

King's Birthday is June in NSW, VIC and SA but October in QLD. Melbourne Cup is
Victoria only; Adelaide Cup is South Australia only.

That means the same calendar date is a public holiday in one city and an ordinary
trading day in another, with everything else held constant. It is the closest thing
to a natural experiment in the dataset, and it is a clean way to test whether a
model attributes a demand spike to the holiday or smears it across correlated
calendar features.

## Data quality

`sales` contains deliberate imperfections, matching the original generator:

- 0.2% outliers (sales multiplied by 3)
- 1.0% missing values (`NaN`)

Because of the `NaN`s, `sales` loads as `float64` rather than `int`.

## Reproducibility

One seeded `numpy` Generator is created at module level and threaded through every
function. Re-running gives byte-identical output.

Note: the original generator called `np.random.seed(None)` inside the store loop,
which reseeds from OS entropy and makes output irreproducible despite the seed at
the top of the file. That is not reproduced here.
