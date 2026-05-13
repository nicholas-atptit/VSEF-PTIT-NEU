
try:
    from datetime import date

    from src.data.providers.vn_provider_contract import AssetType, FetchRequest, Frequency, SourceName
    from src.data.providers.vn_price_gateway import fetch_price_history

    today = date.today().isoformat()
    response = fetch_price_history(
        FetchRequest(
            symbol="FPT",
            asset_type=AssetType.STOCK,
            start=today,
            end=today,
            frequency=Frequency.HOURLY,
            preferred_sources=(SourceName.KBS, SourceName.VCI),
            allow_legacy_fallback=True,
            allow_daily=False,
            allow_resample=False,
        )
    )
    df = response.data
    if not df.empty:
        print(f"LATEST_PRICE_REST: {df.iloc[-1]['close']}")
    else:
        print("REST_EMPTY")
except Exception as e:
    print(f"REST_ERROR: {str(e)}")
