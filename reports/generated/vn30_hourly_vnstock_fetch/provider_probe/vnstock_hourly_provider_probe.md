# vnstock Hourly Provider Probe

## Package Detection

| package | installed | version | origin |
| --- | --- | --- | --- |
| vnstock_data | false |  |  |
| vnstock | true | 3.5.0 | C:\Python\Lib\site-packages\vnstock\__init__.py |

## Probe Decision

- Can provider fetch hourly stock data: True.
- Can provider fetch hourly VNINDEX: True.
- VN30INDEX exact-code support: False.
- VNXALL exact-code support: False.
- Best provider/source/function: vnstock / VCI / Quote.history(interval=1H).
- Success requires actual standardized hourly OHLCV rows, not provider availability claims.

## Successful Attempts

| symbol | sample_start | sample_end | package | source | function_used | standardized_rows | first_timestamp | last_timestamp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACB | 2024-01-02 | 2024-01-05 | vnstock | VCI | Quote.history(interval=1H) | 21 | 2023-12-28 14:00:00 | 2024-01-05 14:00:00 |
| ACB | 2025-01-02 | 2025-01-05 | vnstock | VCI | Quote.history(interval=1H) | 16 | 2024-12-30 10:00:00 | 2025-01-03 14:00:00 |
| ACB | 2026-05-04 | 2026-05-11 | vnstock | VCI | Quote.history(interval=1H) | 36 | 2026-04-28 14:00:00 | 2026-05-11 14:00:00 |
| ACB | 2026-05-04 | 2026-05-11 | vnstock | KBS | Quote.history(interval=1H) | 30 | 2026-05-04 09:00:00 | 2026-05-11 14:00:00 |
| ACB | 2026-05-04 | 2026-05-11 | vnstock | KBS | Vnstock.stock.quote.history(interval=1H) | 30 | 2026-05-04 09:00:00 | 2026-05-11 14:00:00 |
| HPG | 2024-01-02 | 2024-01-05 | vnstock | VCI | Quote.history(interval=1H) | 21 | 2023-12-28 14:00:00 | 2024-01-05 14:00:00 |
| HPG | 2025-01-02 | 2025-01-05 | vnstock | VCI | Quote.history(interval=1H) | 16 | 2024-12-30 10:00:00 | 2025-01-03 14:00:00 |
| HPG | 2026-05-04 | 2026-05-11 | vnstock | VCI | Quote.history(interval=1H) | 36 | 2026-04-28 14:00:00 | 2026-05-11 14:00:00 |
| HPG | 2026-05-04 | 2026-05-11 | vnstock | KBS | Quote.history(interval=1H) | 30 | 2026-05-04 09:00:00 | 2026-05-11 14:00:00 |
| HPG | 2026-05-04 | 2026-05-11 | vnstock | KBS | Vnstock.stock.quote.history(interval=1H) | 30 | 2026-05-04 09:00:00 | 2026-05-11 14:00:00 |
| VNINDEX | 2024-01-02 | 2024-01-05 | vnstock | VCI | Quote.history(interval=1H) | 21 | 2023-12-28 14:00:00 | 2024-01-05 14:00:00 |
| VNINDEX | 2024-01-02 | 2024-01-05 | vnstock | KBS | Quote.history(interval=1H) | 20 | 2024-01-02 09:00:00 | 2024-01-05 14:00:00 |
| VNINDEX | 2024-01-02 | 2024-01-05 | vnstock | VCI | Vnstock.stock.quote.history(interval=1H) | 21 | 2023-12-28 14:00:00 | 2024-01-05 14:00:00 |
| VNINDEX | 2024-01-02 | 2024-01-05 | vnstock | KBS | Vnstock.stock.quote.history(interval=1H) | 20 | 2024-01-02 09:00:00 | 2024-01-05 14:00:00 |
| VNINDEX | 2025-01-02 | 2025-01-05 | vnstock | VCI | Quote.history(interval=1H) | 16 | 2024-12-30 10:00:00 | 2025-01-03 14:00:00 |
| VNINDEX | 2025-01-02 | 2025-01-05 | vnstock | KBS | Quote.history(interval=1H) | 10 | 2025-01-02 09:00:00 | 2025-01-03 14:00:00 |
| VNINDEX | 2025-01-02 | 2025-01-05 | vnstock | VCI | Vnstock.stock.quote.history(interval=1H) | 16 | 2024-12-30 10:00:00 | 2025-01-03 14:00:00 |
| VNINDEX | 2025-01-02 | 2025-01-05 | vnstock | KBS | Vnstock.stock.quote.history(interval=1H) | 10 | 2025-01-02 09:00:00 | 2025-01-03 14:00:00 |
| VNINDEX | 2026-05-04 | 2026-05-11 | vnstock | VCI | Quote.history(interval=1H) | 36 | 2026-04-28 14:00:00 | 2026-05-11 14:00:00 |
| VNINDEX | 2026-05-04 | 2026-05-11 | vnstock | KBS | Quote.history(interval=1H) | 30 | 2026-05-04 09:00:00 | 2026-05-11 14:00:00 |
| VNINDEX | 2026-05-04 | 2026-05-11 | vnstock | VCI | Vnstock.stock.quote.history(interval=1H) | 36 | 2026-04-28 14:00:00 | 2026-05-11 14:00:00 |
| VNINDEX | 2026-05-04 | 2026-05-11 | vnstock | KBS | Vnstock.stock.quote.history(interval=1H) | 30 | 2026-05-04 09:00:00 | 2026-05-11 14:00:00 |

## Attempt Log Preview

| symbol | sample_start | sample_end | package | source | function_used | standardized_rows | success | exception_type | exception_message |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACB | 2024-01-02 | 2024-01-05 | vnstock | VCI | Quote.history(interval=1H) | 21 | true |  |  |
| ACB | 2024-01-02 | 2024-01-05 | vnstock | KBS | Quote.history(interval=1H) | 0 | false | RetryError | RetryError[<Future at 0x1d7e53923f0 state=finished raised ValueError>] |
| ACB | 2024-01-02 | 2024-01-05 | vnstock | VND | Quote.history(interval=1H) | 0 | false | ValueError | Lớp Quote chỉ nhận giá trị tham số source là kbs, vci, msn, dnse, binance, fmp, fmarket. |
| ACB | 2024-01-02 | 2024-01-05 | vnstock | MAS | Quote.history(interval=1H) | 0 | false | ValueError | Lớp Quote chỉ nhận giá trị tham số source là kbs, vci, msn, dnse, binance, fmp, fmarket. |
| ACB | 2024-01-02 | 2024-01-05 | vnstock | VCI | Vnstock.stock.quote.history(interval=1H) | 0 | false | KeyError | 'data' |
| ACB | 2024-01-02 | 2024-01-05 | vnstock | KBS | Vnstock.stock.quote.history(interval=1H) | 0 | false | RetryError | RetryError[<Future at 0x1d7e5396580 state=finished raised ValueError>] |
| ACB | 2024-01-02 | 2024-01-05 | vnstock | VND | Vnstock.stock.quote.history(interval=1H) | 0 | false | ValueError | Supported sources: KBS, VCI, MSN, FMP. Got: VND |
| ACB | 2024-01-02 | 2024-01-05 | vnstock | MAS | Vnstock.stock.quote.history(interval=1H) | 0 | false | ValueError | Supported sources: KBS, VCI, MSN, FMP. Got: MAS |
| ACB | 2025-01-02 | 2025-01-05 | vnstock | VCI | Quote.history(interval=1H) | 16 | true |  |  |
| ACB | 2025-01-02 | 2025-01-05 | vnstock | KBS | Quote.history(interval=1H) | 0 | false | RetryError | RetryError[<Future at 0x1d7e54446e0 state=finished raised ValueError>] |
| ACB | 2025-01-02 | 2025-01-05 | vnstock | VND | Quote.history(interval=1H) | 0 | false | ValueError | Lớp Quote chỉ nhận giá trị tham số source là kbs, vci, msn, dnse, binance, fmp, fmarket. |
| ACB | 2025-01-02 | 2025-01-05 | vnstock | MAS | Quote.history(interval=1H) | 0 | false | ValueError | Lớp Quote chỉ nhận giá trị tham số source là kbs, vci, msn, dnse, binance, fmp, fmarket. |
| ACB | 2025-01-02 | 2025-01-05 | vnstock | VCI | Vnstock.stock.quote.history(interval=1H) | 0 | false | KeyError | 'data' |
| ACB | 2025-01-02 | 2025-01-05 | vnstock | KBS | Vnstock.stock.quote.history(interval=1H) | 0 | false | RetryError | RetryError[<Future at 0x1d7e5471010 state=finished raised ValueError>] |
| ACB | 2025-01-02 | 2025-01-05 | vnstock | VND | Vnstock.stock.quote.history(interval=1H) | 0 | false | ValueError | Supported sources: KBS, VCI, MSN, FMP. Got: VND |
| ACB | 2025-01-02 | 2025-01-05 | vnstock | MAS | Vnstock.stock.quote.history(interval=1H) | 0 | false | ValueError | Supported sources: KBS, VCI, MSN, FMP. Got: MAS |
| ACB | 2026-05-04 | 2026-05-11 | vnstock | VCI | Quote.history(interval=1H) | 36 | true |  |  |
| ACB | 2026-05-04 | 2026-05-11 | vnstock | KBS | Quote.history(interval=1H) | 30 | true |  |  |
| ACB | 2026-05-04 | 2026-05-11 | vnstock | VND | Quote.history(interval=1H) | 0 | false | ValueError | Lớp Quote chỉ nhận giá trị tham số source là kbs, vci, msn, dnse, binance, fmp, fmarket. |
| ACB | 2026-05-04 | 2026-05-11 | vnstock | MAS | Quote.history(interval=1H) | 0 | false | ValueError | Lớp Quote chỉ nhận giá trị tham số source là kbs, vci, msn, dnse, binance, fmp, fmarket. |
| ACB | 2026-05-04 | 2026-05-11 | vnstock | VCI | Vnstock.stock.quote.history(interval=1H) | 0 | false | KeyError | 'data' |
| ACB | 2026-05-04 | 2026-05-11 | vnstock | KBS | Vnstock.stock.quote.history(interval=1H) | 30 | true |  |  |
| ACB | 2026-05-04 | 2026-05-11 | vnstock | VND | Vnstock.stock.quote.history(interval=1H) | 0 | false | ValueError | Supported sources: KBS, VCI, MSN, FMP. Got: VND |
| ACB | 2026-05-04 | 2026-05-11 | vnstock | MAS | Vnstock.stock.quote.history(interval=1H) | 0 | false | ValueError | Supported sources: KBS, VCI, MSN, FMP. Got: MAS |
| HPG | 2024-01-02 | 2024-01-05 | vnstock | VCI | Quote.history(interval=1H) | 21 | true |  |  |
| HPG | 2024-01-02 | 2024-01-05 | vnstock | KBS | Quote.history(interval=1H) | 0 | false | RetryError | RetryError[<Future at 0x1d7e5436e90 state=finished raised ValueError>] |
| HPG | 2024-01-02 | 2024-01-05 | vnstock | VND | Quote.history(interval=1H) | 0 | false | ValueError | Lớp Quote chỉ nhận giá trị tham số source là kbs, vci, msn, dnse, binance, fmp, fmarket. |
| HPG | 2024-01-02 | 2024-01-05 | vnstock | MAS | Quote.history(interval=1H) | 0 | false | ValueError | Lớp Quote chỉ nhận giá trị tham số source là kbs, vci, msn, dnse, binance, fmp, fmarket. |
| HPG | 2024-01-02 | 2024-01-05 | vnstock | VCI | Vnstock.stock.quote.history(interval=1H) | 0 | false | KeyError | 'data' |
| HPG | 2024-01-02 | 2024-01-05 | vnstock | KBS | Vnstock.stock.quote.history(interval=1H) | 0 | false | RetryError | RetryError[<Future at 0x1d7e544f2d0 state=finished raised ValueError>] |
| HPG | 2024-01-02 | 2024-01-05 | vnstock | VND | Vnstock.stock.quote.history(interval=1H) | 0 | false | ValueError | Supported sources: KBS, VCI, MSN, FMP. Got: VND |
| HPG | 2024-01-02 | 2024-01-05 | vnstock | MAS | Vnstock.stock.quote.history(interval=1H) | 0 | false | ValueError | Supported sources: KBS, VCI, MSN, FMP. Got: MAS |
| HPG | 2025-01-02 | 2025-01-05 | vnstock | VCI | Quote.history(interval=1H) | 16 | true |  |  |
| HPG | 2025-01-02 | 2025-01-05 | vnstock | KBS | Quote.history(interval=1H) | 0 | false | RetryError | RetryError[<Future at 0x1d7e544e250 state=finished raised ValueError>] |
| HPG | 2025-01-02 | 2025-01-05 | vnstock | VND | Quote.history(interval=1H) | 0 | false | ValueError | Lớp Quote chỉ nhận giá trị tham số source là kbs, vci, msn, dnse, binance, fmp, fmarket. |
| HPG | 2025-01-02 | 2025-01-05 | vnstock | MAS | Quote.history(interval=1H) | 0 | false | ValueError | Lớp Quote chỉ nhận giá trị tham số source là kbs, vci, msn, dnse, binance, fmp, fmarket. |
| HPG | 2025-01-02 | 2025-01-05 | vnstock | VCI | Vnstock.stock.quote.history(interval=1H) | 0 | false | KeyError | 'data' |
| HPG | 2025-01-02 | 2025-01-05 | vnstock | KBS | Vnstock.stock.quote.history(interval=1H) | 0 | false | RetryError | RetryError[<Future at 0x1d7e544f8d0 state=finished raised ValueError>] |
| HPG | 2025-01-02 | 2025-01-05 | vnstock | VND | Vnstock.stock.quote.history(interval=1H) | 0 | false | ValueError | Supported sources: KBS, VCI, MSN, FMP. Got: VND |
| HPG | 2025-01-02 | 2025-01-05 | vnstock | MAS | Vnstock.stock.quote.history(interval=1H) | 0 | false | ValueError | Supported sources: KBS, VCI, MSN, FMP. Got: MAS |
| HPG | 2026-05-04 | 2026-05-11 | vnstock | VCI | Quote.history(interval=1H) | 36 | true |  |  |
| HPG | 2026-05-04 | 2026-05-11 | vnstock | KBS | Quote.history(interval=1H) | 30 | true |  |  |
| HPG | 2026-05-04 | 2026-05-11 | vnstock | VND | Quote.history(interval=1H) | 0 | false | ValueError | Lớp Quote chỉ nhận giá trị tham số source là kbs, vci, msn, dnse, binance, fmp, fmarket. |
| HPG | 2026-05-04 | 2026-05-11 | vnstock | MAS | Quote.history(interval=1H) | 0 | false | ValueError | Lớp Quote chỉ nhận giá trị tham số source là kbs, vci, msn, dnse, binance, fmp, fmarket. |
| HPG | 2026-05-04 | 2026-05-11 | vnstock | VCI | Vnstock.stock.quote.history(interval=1H) | 0 | false | KeyError | 'data' |
| HPG | 2026-05-04 | 2026-05-11 | vnstock | KBS | Vnstock.stock.quote.history(interval=1H) | 30 | true |  |  |
| HPG | 2026-05-04 | 2026-05-11 | vnstock | VND | Vnstock.stock.quote.history(interval=1H) | 0 | false | ValueError | Supported sources: KBS, VCI, MSN, FMP. Got: VND |
| HPG | 2026-05-04 | 2026-05-11 | vnstock | MAS | Vnstock.stock.quote.history(interval=1H) | 0 | false | ValueError | Supported sources: KBS, VCI, MSN, FMP. Got: MAS |
| VNINDEX | 2024-01-02 | 2024-01-05 | vnstock | VCI | Quote.history(interval=1H) | 21 | true |  |  |
| VNINDEX | 2024-01-02 | 2024-01-05 | vnstock | KBS | Quote.history(interval=1H) | 20 | true |  |  |
| VNINDEX | 2024-01-02 | 2024-01-05 | vnstock | VND | Quote.history(interval=1H) | 0 | false | ValueError | Lớp Quote chỉ nhận giá trị tham số source là kbs, vci, msn, dnse, binance, fmp, fmarket. |
| VNINDEX | 2024-01-02 | 2024-01-05 | vnstock | MAS | Quote.history(interval=1H) | 0 | false | ValueError | Lớp Quote chỉ nhận giá trị tham số source là kbs, vci, msn, dnse, binance, fmp, fmarket. |
| VNINDEX | 2024-01-02 | 2024-01-05 | vnstock | VCI | Vnstock.stock.quote.history(interval=1H) | 21 | true |  |  |
| VNINDEX | 2024-01-02 | 2024-01-05 | vnstock | KBS | Vnstock.stock.quote.history(interval=1H) | 20 | true |  |  |
| VNINDEX | 2024-01-02 | 2024-01-05 | vnstock | VND | Vnstock.stock.quote.history(interval=1H) | 0 | false | ValueError | Supported sources: KBS, VCI, MSN, FMP. Got: VND |
| VNINDEX | 2024-01-02 | 2024-01-05 | vnstock | MAS | Vnstock.stock.quote.history(interval=1H) | 0 | false | ValueError | Supported sources: KBS, VCI, MSN, FMP. Got: MAS |
| VNINDEX | 2025-01-02 | 2025-01-05 | vnstock | VCI | Quote.history(interval=1H) | 16 | true |  |  |
| VNINDEX | 2025-01-02 | 2025-01-05 | vnstock | KBS | Quote.history(interval=1H) | 10 | true |  |  |
| VNINDEX | 2025-01-02 | 2025-01-05 | vnstock | VND | Quote.history(interval=1H) | 0 | false | ValueError | Lớp Quote chỉ nhận giá trị tham số source là kbs, vci, msn, dnse, binance, fmp, fmarket. |
| VNINDEX | 2025-01-02 | 2025-01-05 | vnstock | MAS | Quote.history(interval=1H) | 0 | false | ValueError | Lớp Quote chỉ nhận giá trị tham số source là kbs, vci, msn, dnse, binance, fmp, fmarket. |
| VNINDEX | 2025-01-02 | 2025-01-05 | vnstock | VCI | Vnstock.stock.quote.history(interval=1H) | 16 | true |  |  |
| VNINDEX | 2025-01-02 | 2025-01-05 | vnstock | KBS | Vnstock.stock.quote.history(interval=1H) | 10 | true |  |  |
| VNINDEX | 2025-01-02 | 2025-01-05 | vnstock | VND | Vnstock.stock.quote.history(interval=1H) | 0 | false | ValueError | Supported sources: KBS, VCI, MSN, FMP. Got: VND |
| VNINDEX | 2025-01-02 | 2025-01-05 | vnstock | MAS | Vnstock.stock.quote.history(interval=1H) | 0 | false | ValueError | Supported sources: KBS, VCI, MSN, FMP. Got: MAS |
| VNINDEX | 2026-05-04 | 2026-05-11 | vnstock | VCI | Quote.history(interval=1H) | 36 | true |  |  |
| VNINDEX | 2026-05-04 | 2026-05-11 | vnstock | KBS | Quote.history(interval=1H) | 30 | true |  |  |
| VNINDEX | 2026-05-04 | 2026-05-11 | vnstock | VND | Quote.history(interval=1H) | 0 | false | ValueError | Lớp Quote chỉ nhận giá trị tham số source là kbs, vci, msn, dnse, binance, fmp, fmarket. |
| VNINDEX | 2026-05-04 | 2026-05-11 | vnstock | MAS | Quote.history(interval=1H) | 0 | false | ValueError | Lớp Quote chỉ nhận giá trị tham số source là kbs, vci, msn, dnse, binance, fmp, fmarket. |
| VNINDEX | 2026-05-04 | 2026-05-11 | vnstock | VCI | Vnstock.stock.quote.history(interval=1H) | 36 | true |  |  |
| VNINDEX | 2026-05-04 | 2026-05-11 | vnstock | KBS | Vnstock.stock.quote.history(interval=1H) | 30 | true |  |  |
| VNINDEX | 2026-05-04 | 2026-05-11 | vnstock | VND | Vnstock.stock.quote.history(interval=1H) | 0 | false | ValueError | Supported sources: KBS, VCI, MSN, FMP. Got: VND |
| VNINDEX | 2026-05-04 | 2026-05-11 | vnstock | MAS | Vnstock.stock.quote.history(interval=1H) | 0 | false | ValueError | Supported sources: KBS, VCI, MSN, FMP. Got: MAS |
| VN30INDEX | 2024-01-02 | 2024-01-05 | vnstock | VCI | Quote.history(interval=1H) | 0 | false | ValueError | Invalid derivative or bond symbol. Symbol must be in format of VN30F1M, VN30F2024, GB10F2024, or for company bonds, e.g., BAB122032 |
| VN30INDEX | 2024-01-02 | 2024-01-05 | vnstock | KBS | Quote.history(interval=1H) | 0 | false | ValueError | Invalid derivative or bond symbol. Symbol must be in format of VN30F1M, VN30F2024, GB10F2024, or for company bonds, e.g., BAB122032 |
| VN30INDEX | 2024-01-02 | 2024-01-05 | vnstock | VND | Quote.history(interval=1H) | 0 | false | ValueError | Lớp Quote chỉ nhận giá trị tham số source là kbs, vci, msn, dnse, binance, fmp, fmarket. |
| VN30INDEX | 2024-01-02 | 2024-01-05 | vnstock | MAS | Quote.history(interval=1H) | 0 | false | ValueError | Lớp Quote chỉ nhận giá trị tham số source là kbs, vci, msn, dnse, binance, fmp, fmarket. |
| VN30INDEX | 2024-01-02 | 2024-01-05 | vnstock | VCI | Vnstock.stock.quote.history(interval=1H) | 0 | false | ValueError | Invalid derivative or bond symbol. Symbol must be in format of VN30F1M, VN30F2024, GB10F2024, or for company bonds, e.g., BAB122032 |
| VN30INDEX | 2024-01-02 | 2024-01-05 | vnstock | KBS | Vnstock.stock.quote.history(interval=1H) | 0 | false | ValueError | Invalid derivative or bond symbol. Symbol must be in format of VN30F1M, VN30F2024, GB10F2024, or for company bonds, e.g., BAB122032 |
| VN30INDEX | 2024-01-02 | 2024-01-05 | vnstock | VND | Vnstock.stock.quote.history(interval=1H) | 0 | false | ValueError | Supported sources: KBS, VCI, MSN, FMP. Got: VND |
| VN30INDEX | 2024-01-02 | 2024-01-05 | vnstock | MAS | Vnstock.stock.quote.history(interval=1H) | 0 | false | ValueError | Supported sources: KBS, VCI, MSN, FMP. Got: MAS |
| VN30INDEX | 2025-01-02 | 2025-01-05 | vnstock | VCI | Quote.history(interval=1H) | 0 | false | ValueError | Invalid derivative or bond symbol. Symbol must be in format of VN30F1M, VN30F2024, GB10F2024, or for company bonds, e.g., BAB122032 |
| VN30INDEX | 2025-01-02 | 2025-01-05 | vnstock | KBS | Quote.history(interval=1H) | 0 | false | ValueError | Invalid derivative or bond symbol. Symbol must be in format of VN30F1M, VN30F2024, GB10F2024, or for company bonds, e.g., BAB122032 |
| VN30INDEX | 2025-01-02 | 2025-01-05 | vnstock | VND | Quote.history(interval=1H) | 0 | false | ValueError | Lớp Quote chỉ nhận giá trị tham số source là kbs, vci, msn, dnse, binance, fmp, fmarket. |
| VN30INDEX | 2025-01-02 | 2025-01-05 | vnstock | MAS | Quote.history(interval=1H) | 0 | false | ValueError | Lớp Quote chỉ nhận giá trị tham số source là kbs, vci, msn, dnse, binance, fmp, fmarket. |
| VN30INDEX | 2025-01-02 | 2025-01-05 | vnstock | VCI | Vnstock.stock.quote.history(interval=1H) | 0 | false | ValueError | Invalid derivative or bond symbol. Symbol must be in format of VN30F1M, VN30F2024, GB10F2024, or for company bonds, e.g., BAB122032 |
| VN30INDEX | 2025-01-02 | 2025-01-05 | vnstock | KBS | Vnstock.stock.quote.history(interval=1H) | 0 | false | ValueError | Invalid derivative or bond symbol. Symbol must be in format of VN30F1M, VN30F2024, GB10F2024, or for company bonds, e.g., BAB122032 |
| VN30INDEX | 2025-01-02 | 2025-01-05 | vnstock | VND | Vnstock.stock.quote.history(interval=1H) | 0 | false | ValueError | Supported sources: KBS, VCI, MSN, FMP. Got: VND |
| VN30INDEX | 2025-01-02 | 2025-01-05 | vnstock | MAS | Vnstock.stock.quote.history(interval=1H) | 0 | false | ValueError | Supported sources: KBS, VCI, MSN, FMP. Got: MAS |
| VN30INDEX | 2026-05-04 | 2026-05-11 | vnstock | VCI | Quote.history(interval=1H) | 0 | false | ValueError | Invalid derivative or bond symbol. Symbol must be in format of VN30F1M, VN30F2024, GB10F2024, or for company bonds, e.g., BAB122032 |
| VN30INDEX | 2026-05-04 | 2026-05-11 | vnstock | KBS | Quote.history(interval=1H) | 0 | false | ValueError | Invalid derivative or bond symbol. Symbol must be in format of VN30F1M, VN30F2024, GB10F2024, or for company bonds, e.g., BAB122032 |
| VN30INDEX | 2026-05-04 | 2026-05-11 | vnstock | VND | Quote.history(interval=1H) | 0 | false | ValueError | Lớp Quote chỉ nhận giá trị tham số source là kbs, vci, msn, dnse, binance, fmp, fmarket. |
| VN30INDEX | 2026-05-04 | 2026-05-11 | vnstock | MAS | Quote.history(interval=1H) | 0 | false | ValueError | Lớp Quote chỉ nhận giá trị tham số source là kbs, vci, msn, dnse, binance, fmp, fmarket. |
| VN30INDEX | 2026-05-04 | 2026-05-11 | vnstock | VCI | Vnstock.stock.quote.history(interval=1H) | 0 | false | ValueError | Invalid derivative or bond symbol. Symbol must be in format of VN30F1M, VN30F2024, GB10F2024, or for company bonds, e.g., BAB122032 |
| VN30INDEX | 2026-05-04 | 2026-05-11 | vnstock | KBS | Vnstock.stock.quote.history(interval=1H) | 0 | false | ValueError | Invalid derivative or bond symbol. Symbol must be in format of VN30F1M, VN30F2024, GB10F2024, or for company bonds, e.g., BAB122032 |
| VN30INDEX | 2026-05-04 | 2026-05-11 | vnstock | VND | Vnstock.stock.quote.history(interval=1H) | 0 | false | ValueError | Supported sources: KBS, VCI, MSN, FMP. Got: VND |
| VN30INDEX | 2026-05-04 | 2026-05-11 | vnstock | MAS | Vnstock.stock.quote.history(interval=1H) | 0 | false | ValueError | Supported sources: KBS, VCI, MSN, FMP. Got: MAS |
| VNXALL | 2024-01-02 | 2024-01-05 | vnstock | VCI | Quote.history(interval=1H) | 0 | false | RetryError | RetryError[<Future at 0x1d7b2d91850 state=finished raised ValueError>] |
| VNXALL | 2024-01-02 | 2024-01-05 | vnstock | KBS | Quote.history(interval=1H) | 0 | false | ValueError | Mã chỉ số 'VNXALL' không được hỗ trợ bởi KBS. Các chỉ số hợp lệ: VNINDEX, HNXINDEX, UPCOMINDEX, VN30, HNX30, VN100 |
| VNXALL | 2024-01-02 | 2024-01-05 | vnstock | VND | Quote.history(interval=1H) | 0 | false | ValueError | Lớp Quote chỉ nhận giá trị tham số source là kbs, vci, msn, dnse, binance, fmp, fmarket. |
| VNXALL | 2024-01-02 | 2024-01-05 | vnstock | MAS | Quote.history(interval=1H) | 0 | false | ValueError | Lớp Quote chỉ nhận giá trị tham số source là kbs, vci, msn, dnse, binance, fmp, fmarket. |
| VNXALL | 2024-01-02 | 2024-01-05 | vnstock | VCI | Vnstock.stock.quote.history(interval=1H) | 0 | false | RetryError | RetryError[<Future at 0x1d7e544e550 state=finished raised ValueError>] |
| VNXALL | 2024-01-02 | 2024-01-05 | vnstock | KBS | Vnstock.stock.quote.history(interval=1H) | 0 | false | ValueError | Mã chỉ số 'VNXALL' không được hỗ trợ bởi KBS. Các chỉ số hợp lệ: VNINDEX, HNXINDEX, UPCOMINDEX, VN30, HNX30, VN100 |
| VNXALL | 2024-01-02 | 2024-01-05 | vnstock | VND | Vnstock.stock.quote.history(interval=1H) | 0 | false | ValueError | Supported sources: KBS, VCI, MSN, FMP. Got: VND |
| VNXALL | 2024-01-02 | 2024-01-05 | vnstock | MAS | Vnstock.stock.quote.history(interval=1H) | 0 | false | ValueError | Supported sources: KBS, VCI, MSN, FMP. Got: MAS |
| VNXALL | 2025-01-02 | 2025-01-05 | vnstock | VCI | Quote.history(interval=1H) | 0 | false | RetryError | RetryError[<Future at 0x1d7e53cd650 state=finished raised ValueError>] |
| VNXALL | 2025-01-02 | 2025-01-05 | vnstock | KBS | Quote.history(interval=1H) | 0 | false | ValueError | Mã chỉ số 'VNXALL' không được hỗ trợ bởi KBS. Các chỉ số hợp lệ: VNINDEX, HNXINDEX, UPCOMINDEX, VN30, HNX30, VN100 |
| VNXALL | 2025-01-02 | 2025-01-05 | vnstock | VND | Quote.history(interval=1H) | 0 | false | ValueError | Lớp Quote chỉ nhận giá trị tham số source là kbs, vci, msn, dnse, binance, fmp, fmarket. |
| VNXALL | 2025-01-02 | 2025-01-05 | vnstock | MAS | Quote.history(interval=1H) | 0 | false | ValueError | Lớp Quote chỉ nhận giá trị tham số source là kbs, vci, msn, dnse, binance, fmp, fmarket. |
| VNXALL | 2025-01-02 | 2025-01-05 | vnstock | VCI | Vnstock.stock.quote.history(interval=1H) | 0 | false | RetryError | RetryError[<Future at 0x1d7e5481bd0 state=finished raised ValueError>] |
| VNXALL | 2025-01-02 | 2025-01-05 | vnstock | KBS | Vnstock.stock.quote.history(interval=1H) | 0 | false | ValueError | Mã chỉ số 'VNXALL' không được hỗ trợ bởi KBS. Các chỉ số hợp lệ: VNINDEX, HNXINDEX, UPCOMINDEX, VN30, HNX30, VN100 |
| VNXALL | 2025-01-02 | 2025-01-05 | vnstock | VND | Vnstock.stock.quote.history(interval=1H) | 0 | false | ValueError | Supported sources: KBS, VCI, MSN, FMP. Got: VND |
| VNXALL | 2025-01-02 | 2025-01-05 | vnstock | MAS | Vnstock.stock.quote.history(interval=1H) | 0 | false | ValueError | Supported sources: KBS, VCI, MSN, FMP. Got: MAS |
| VNXALL | 2026-05-04 | 2026-05-11 | vnstock | VCI | Quote.history(interval=1H) | 0 | false | RetryError | RetryError[<Future at 0x1d7e5483550 state=finished raised ValueError>] |
| VNXALL | 2026-05-04 | 2026-05-11 | vnstock | KBS | Quote.history(interval=1H) | 0 | false | ValueError | Mã chỉ số 'VNXALL' không được hỗ trợ bởi KBS. Các chỉ số hợp lệ: VNINDEX, HNXINDEX, UPCOMINDEX, VN30, HNX30, VN100 |
| VNXALL | 2026-05-04 | 2026-05-11 | vnstock | VND | Quote.history(interval=1H) | 0 | false | ValueError | Lớp Quote chỉ nhận giá trị tham số source là kbs, vci, msn, dnse, binance, fmp, fmarket. |
| VNXALL | 2026-05-04 | 2026-05-11 | vnstock | MAS | Quote.history(interval=1H) | 0 | false | ValueError | Lớp Quote chỉ nhận giá trị tham số source là kbs, vci, msn, dnse, binance, fmp, fmarket. |
| VNXALL | 2026-05-04 | 2026-05-11 | vnstock | VCI | Vnstock.stock.quote.history(interval=1H) | 0 | false | RetryError | RetryError[<Future at 0x1d7e5481650 state=finished raised ValueError>] |
| VNXALL | 2026-05-04 | 2026-05-11 | vnstock | KBS | Vnstock.stock.quote.history(interval=1H) | 0 | false | ValueError | Mã chỉ số 'VNXALL' không được hỗ trợ bởi KBS. Các chỉ số hợp lệ: VNINDEX, HNXINDEX, UPCOMINDEX, VN30, HNX30, VN100 |
| VNXALL | 2026-05-04 | 2026-05-11 | vnstock | VND | Vnstock.stock.quote.history(interval=1H) | 0 | false | ValueError | Supported sources: KBS, VCI, MSN, FMP. Got: VND |
| VNXALL | 2026-05-04 | 2026-05-11 | vnstock | MAS | Vnstock.stock.quote.history(interval=1H) | 0 | false | ValueError | Supported sources: KBS, VCI, MSN, FMP. Got: MAS |
