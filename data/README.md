# Data

## Source

**Vehicle dataset from CarDekho**, published on Kaggle by the user `nehalbirla`:
<https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho>

This project uses the file **`car data.csv`** (301 rows, 9 columns): used-vehicle listings
scraped from [CarDekho.com](https://www.cardekho.com/), an Indian vehicle marketplace.

**License:** see the dataset's Kaggle page for its current license and terms of use. The CSV is
**not redistributed in this repository** (it is git-ignored). Download it yourself with one of
the options below.

## Download

### Option A: script (recommended)

```bash
python -m src.download_data
```

This fetches a public GitHub mirror of the identical file and **verifies its SHA-256 checksum**
(`00b477de…4b3063`), so you train on exactly the bytes this project was evaluated on. A
mismatch is rejected rather than silently used.

### Option B: Kaggle

1. Sign in to Kaggle and download the dataset from the link above, or use the Kaggle CLI:
   `kaggle datasets download -d nehalbirla/vehicle-dataset-from-cardekho --unzip`
2. Copy **`car data.csv`** to **`data/raw/car_data.csv`** (note the underscore).
3. Optionally run `python -m src.download_data` to verify the checksum.

You can also point the code at any location with the `CAR_DATA_PATH` environment variable.

## Schema

| Column | Type | Description | Used as |
|---|---|---|---|
| `Car_Name` | text | Model name (98 unique values) | Cleaning only (two-wheeler detection) |
| `Year` | int | Year of manufacture (2003 to 2018) | Feature → `Car_Age` |
| `Selling_Price` | float | Price the vehicle was listed/sold for, **lakh INR** | **Target** |
| `Present_Price` | float | Current ex-showroom price of the same model, **lakh INR** | Feature |
| `Kms_Driven` | int | Odometer reading in km | Feature |
| `Fuel_Type` | text | Petrol / Diesel / CNG | Feature (one-hot) |
| `Seller_Type` | text | Dealer / Individual | Feature (one-hot) |
| `Transmission` | text | Manual / Automatic | Feature (one-hot) |
| `Owner` | int | Number of previous owners (0, 1 or 3) | Feature |

1 lakh = ₹1,00,000 = 100,000 INR.

## Data-quality findings

| Check | Result | Action |
|---|---|---|
| Missing values | 0 | none needed (the cleaning code still handles them) |
| Exact duplicate rows | 2 | removed *before* splitting, so one copy cannot sit in train and the other in test |
| Two-wheelers | **101 rows** (Bajaj, Hero, Royal Enfield, KTM, Activa, …), one contiguous block (rows 100 to 200) | removed: out of scope for a *car* price model |
| Extreme mileage | 500,000 km (an Activa scooter) | removed together with the two-wheelers |
| Rare categories | CNG: 2 cars; Individual seller: 5 cars; Owner > 0: 5 cars | kept; documented as a limitation |
| Price outliers | 13 IQR outliers in the training split, e.g. a 35-lakh Land Cruiser | kept: they are genuine vehicles |

After cleaning: **198 cars** (158 train / 40 test).
