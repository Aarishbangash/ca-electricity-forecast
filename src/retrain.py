def update_dataset(api_key: str) -> bool:

    # --- Define last 5 days ---
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=4)
    date_range = pd.date_range(start=start_date, end=end_date)

    print(f"\n── STEP 1: Fetch last 5 days ────────────────────")
    print(f"  Range: {start_date.date()} → {end_date.date()}")

    all_new_data = []

    # --- Loop through each day ---
    for date in date_range:
        date_str = date.strftime("%Y-%m-%d")
        print(f"\n  Fetching {date_str}...")

        # =========================
        # EIA FETCH (with retries)
        # =========================
        eia_df = None
        for attempt in range(1, 4):
            eia_df = fetch_eia_day(api_key, date_str)
            if eia_df is not None:
                break
            print(f"    EIA retry {attempt}/3...")
            time.sleep(10)

        if eia_df is None:
            print(f"    Skipping {date_str} (EIA failed)")
            continue

        # =========================
        # WEATHER FETCH
        # =========================
        weather_df = fetch_weather_archive(date_str)
        if weather_df is None:
            print(f"    Skipping {date_str} (Weather failed)")
            continue

        # =========================
        # NORMALIZE TIMESTAMPS
        # =========================
        eia_df["timestamp"] = pd.to_datetime(
            eia_df["timestamp"]
        ).dt.tz_convert(TIMEZONE)

        weather_df["timestamp"] = pd.to_datetime(
            weather_df["timestamp"]
        ).dt.tz_convert(TIMEZONE)

        # =========================
        # MERGE
        # =========================
        merged = pd.merge(
            eia_df, weather_df, on="timestamp", how="inner"
        )

        print(f"    Rows: {len(merged)}")

        all_new_data.append(merged)

    # =========================
    # CHECK IF ANY DATA FETCHED
    # =========================
    if not all_new_data:
        print("  ERROR: No data fetched for any day")
        return False

    new_data = pd.concat(all_new_data, ignore_index=True)
    print(f"\n  Total new rows: {len(new_data)}")

    # =========================
    # LOAD EXISTING DATA
    # =========================
    existing = pd.read_csv(DATA_PATH)
    existing["timestamp"] = pd.to_datetime(
        existing["timestamp"], utc=True
    ).dt.tz_convert(TIMEZONE)

    # =========================
    # APPEND + REMOVE DUPLICATES
    # =========================
    combined = (
        pd.concat([existing, new_data], ignore_index=True)
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # =========================
    # ROLLING 5-YEAR WINDOW
    # =========================
    cutoff = (
        pd.Timestamp.now(tz=TIMEZONE)
        - pd.DateOffset(years=5)
    )

    combined = combined[
        combined["timestamp"] >= cutoff
    ].reset_index(drop=True)

    # =========================
    # SAVE
    # =========================
    combined.to_csv(DATA_PATH, index=False)

    print(f"  Dataset: {len(combined):,} rows")
    print(
        f"  Range  : {combined.timestamp.min().date()} "
        f"→ {combined.timestamp.max().date()}"
    )

    return True