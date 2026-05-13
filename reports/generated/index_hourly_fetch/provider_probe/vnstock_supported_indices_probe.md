# vnstock Supported Indices Hourly Probe

- interpreter: `<repo-approved-venv-python>`
- intended venv used: yes
- vnstock_data importable: yes
- vnstock importable: yes
- vnstock version: `NO __version__`
- interval: `1H`

## Probe Summary

| index_code | any hourly rows | rows in 2024 | rows in 2025 | rows in 2026 | provider | source |
|---|---:|---:|---:|---:|---|---|
| `VNINDEX` | yes | yes | yes | yes | `vnstock_data` | `KBS` |
| `HNXINDEX` | yes | yes | yes | yes | `vnstock_data` | `KBS` |
| `UPCOMINDEX` | yes | yes | yes | yes | `vnstock_data` | `KBS` |
| `VN30` | yes | yes | yes | yes | `vnstock_data` | `KBS` |
| `HNX30` | yes | yes | yes | yes | `vnstock_data` | `KBS` |
| `VN100` | yes | yes | yes | yes | `vnstock_data` | `VCI` |

## Notes

- This probe is index-only.
- It uses hourly requests only.
- It does not treat sample support as full-history support.
