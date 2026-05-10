"""Legacy module.
Retained for historical compatibility or migration reference.
Not part of canonical governed runtime.
"""

import pandas as pd
import os

def fetch_data():
    output_csv = "reports/news_keywords_baseline.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    company_data = {
        # --- Viettel Group ---
        "VGI": ("Tổng công ty cổ phần Đầu tư Quốc tế Viettel", "Viettel Global"),
        "VTK": ("Công ty cổ phần Tư vấn Thiết kế Viettel", "Viettel Consultant"),
        "VTP": ("Tổng công ty cổ phần Bưu chính Viettel", "Viettel Post"),
        "CTR": ("Tổng công ty cổ phần Công trình Viettel", "Viettel Construction"),
        
        # --- VN30 ---
        "ACB": ("Ngân hàng TMCP Á Châu", "ACB"),
        "BCM": ("Tổng Công ty Đầu tư và Phát triển Công nghiệp", "Becamex IDC"),
        "BID": ("Ngân hàng TMCP Đầu tư và Phát triển Việt Nam", "BIDV"),
        "BVH": ("Tập đoàn Bảo Việt", "Bảo Việt"),
        "CTG": ("Ngân hàng TMCP Công Thương Việt Nam", "VietinBank"),
        "FPT": ("Công ty cổ phần FPT", "Tập đoàn FPT"),
        "GAS": ("Tổng công ty Khí Việt Nam", "PV GAS"),
        "GVR": ("Tập đoàn Công nghiệp Cao su Việt Nam", "Cao su Việt Nam"),
        "HDB": ("Ngân hàng TMCP Phát triển TP. Hồ Chí Minh", "HDBank"),
        "HPG": ("Công ty cổ phần Tập đoàn Hòa Phát", "Hòa Phát"),
        "MBB": ("Ngân hàng TMCP Quân Đội", "MBBank"),
        "MSN": ("Công ty cổ phần Tập đoàn Masan", "Masan Group"),
        "MWG": ("Công ty cổ phần Đầu tư Thế Giới Di Động", "Thế Giới Di Động"),
        "PLX": ("Tập đoàn Xăng dầu Việt Nam", "Petrolimex"),
        "POW": ("Tổng công ty Điện lực Dầu khí Việt Nam", "PV Power"),
        "SAB": ("Tổng công ty cổ phần Bia - Rượu - Nước giải khát Sài Gòn", "Sabeco"),
        "SHB": ("Ngân hàng TMCP Sài Gòn - Hà Nội", "SHB"),
        "SSB": ("Ngân hàng TMCP Đông Nam Á", "SeABank"),
        "SSI": ("Công ty cổ phần Chứng khoán SSI", "Chứng khoán SSI"),
        "STB": ("Ngân hàng TMCP Sài Gòn Thương Tín", "Sacombank"),
        "TCB": ("Ngân hàng TMCP Kỹ Thương Việt Nam", "Techcombank"),
        "TPB": ("Ngân hàng TMCP Tiên Phong", "TPBank"),
        "VCB": ("Ngân hàng TMCP Ngoại thương Việt Nam", "Vietcombank"),
        "VHM": ("Công ty cổ phần Vinhomes", "Vinhomes"),
        "VIB": ("Ngân hàng TMCP Quốc tế Việt Nam", "VIB"),
        "VIC": ("Tập đoàn Vingroup", "Vingroup"),
        "VJC": ("Công ty cổ phần Hàng không Vietjet", "Vietjet Air"),
        "VNM": ("Công ty cổ phần Sữa Việt Nam", "Vinamilk"),
        "VPB": ("Ngân hàng TMCP Việt Nam Thịnh Vượng", "VPBank"),
        "VRE": ("Công ty cổ phần Vincom Retail", "Vincom Retail"),

        # --- VNMidCap (Rest of VN100) ---
        "AAA": ("Công ty cổ phần Nhựa An Phát Xanh", "Nhựa An Phát Xanh"),
        "ANV": ("Công ty cổ phần Nam Việt", "Thủy sản Nam Việt"),
        "ASM": ("Công ty cổ phần Tập đoàn Sao Mai", "Tập đoàn Sao Mai"),
        "BAF": ("Công ty Cổ phần Nông nghiệp BAF Việt Nam", "BAF Việt Nam"),
        "BCG": ("Công ty cổ phần Tập đoàn Bamboo Capital", "Bamboo Capital"),
        "BMI": ("Tổng Công ty cổ phần Bảo Minh", "Bảo Minh"),
        "BMP": ("Công ty cổ phần Nhựa Bình Minh", "Nhựa Bình Minh"),
        "BWE": ("Công ty Cổ phần Nước - Môi trường Bình Dương", "Biwase"),
        "CII": ("Công ty cổ phần Đầu tư Kỹ thuật Hạ tầng TP.HCM", "CII"),
        "CMG": ("Công ty cổ phần Tập đoàn Công nghệ CMC", "Tập đoàn CMC"),
        "CTD": ("Công ty cổ phần Xây dựng Coteccons", "Coteccons"),
        "DBC": ("Công ty cổ phần Tập đoàn Dabaco Việt Nam", "Dabaco"),
        "DGC": ("Công ty cổ phần Tập đoàn Hóa chất Đức Giang", "Hóa chất Đức Giang"),
        "DGW": ("Công ty cổ phần Thế Giới Số", "Digiworld"),
        "DIG": ("Tổng công ty cổ phần Đầu tư Phát triển Xây dựng", "DIC Corp"),
        "DPM": ("Tổng công ty Phân bón và Hóa chất Dầu khí", "Đạm Phú Mỹ"),
        "DPR": ("Công ty cổ phần Cao su Đồng Phú", "Cao su Đồng Phú"),
        "DXG": ("Công ty cổ phần Tập đoàn Đất Xanh", "Đất Xanh"),
        "EIB": ("Ngân hàng TMCP Xuất Nhập Khẩu Việt Nam", "Eximbank"),
        "EVF": ("Công ty Tài chính Cổ phần Điện lực", "EVNFinance"),
        "FCN": ("Công ty cổ phần FECON", "FECON"),
        "FRT": ("Công ty cổ phần Bán lẻ Kỹ thuật số FPT", "FPT Retail"),
        "FTS": ("Công ty cổ phần Chứng khoán FPT", "Chứng khoán FPT"),
        "GEX": ("Công ty cổ phần Tập đoàn GELEX", "GELEX"),
        "GIL": ("Công ty cổ phần Sản xuất Kinh doanh Xuất nhập khẩu Bình Thạnh", "Gilimex"),
        "GMD": ("Công ty cổ phần Gemadept", "Gemadept"),
        "HAG": ("Công ty cổ phần Hoàng Anh Gia Lai", "Hoàng Anh Gia Lai"),
        "HAH": ("Công ty cổ phần Vận tải và Xếp dỡ Hải An", "Hải An"),
        "HCM": ("Công ty cổ phần Chứng khoán TP.HCM", "HSC"),
        "HDC": ("Công ty cổ phần Phát triển Nhà Bà Rịa - Vũng Tàu", "Hodeco"),
        "HDG": ("Công ty cổ phần Tập đoàn Hà Đô", "Tập đoàn Hà Đô"),
        "HHV": ("Công ty Cổ phần Đầu tư Hạ tầng Giao thông Đèo Cả", "Giao thông Đèo Cả"),
        "HSG": ("Công ty cổ phần Tập đoàn Hoa Sen", "Tập đoàn Hoa Sen"),
        "HT1": ("Công ty cổ phần Xi măng Vicem Hà Tiên", "Xi măng Hà Tiên"),
        "IJC": ("Công ty cổ phần Phát triển Hạ tầng Kỹ thuật", "Becamex IJC"),
        "KBC": ("Tổng công ty Phát triển Đô thị Kinh Bắc", "Kinh Bắc"),
        "KDC": ("Công ty cổ phần Tập đoàn KIDO", "Tập đoàn KIDO"),
        "KDH": ("Công ty cổ phần Đầu tư và Kinh doanh Nhà Khang Điền", "Khang Điền"),
        "LCG": ("Công ty cổ phần LIZEN", "Lizen"),
        "LPB": ("Ngân hàng TMCP Lộc Phát Việt Nam", "LPBank"),
        "MSB": ("Ngân hàng TMCP Hàng Hải Việt Nam", "MSB"),
        "NKG": ("Công ty cổ phần Thép Nam Kim", "Thép Nam Kim"),
        "NLG": ("Công ty cổ phần Đầu tư Nam Long", "Nam Long"),
        "NT2": ("Công ty cổ phần Điện lực Dầu khí Nhơn Trạch 2", "Nhơn Trạch 2"),
        "NVL": ("Công ty cổ phần Tập đoàn Đầu tư Địa ốc No Va", "Novaland"),
        "OCB": ("Ngân hàng TMCP Phương Đông", "OCB"),
        "PAN": ("Công ty cổ phần Tập đoàn PAN", "Tập đoàn PAN"),
        "PC1": ("Công ty cổ phần Tập đoàn PC1", "Tập đoàn PC1"),
        "PDR": ("Công ty cổ phần Phát triển Bất động sản Phát Đạt", "Phát Đạt"),
        "PHR": ("Công ty cổ phần Cao su Phước Hòa", "Cao su Phước Hòa"),
        "PNJ": ("Công ty cổ phần Vàng bạc Đá quý Phú Nhuận", "PNJ"),
        "PTB": ("Công ty cổ phần Phú Tài", "Phú Tài"),
        "PVD": ("Tổng công ty cổ phần Khoan và Dịch vụ Khoan Dầu khí", "PV Drilling"),
        "REE": ("Công ty cổ phần Cơ Điện Lạnh", "REE"),
        "SBT": ("Công ty cổ phần Thành Thành Công - Biên Hòa", "TTC AgriS"),
        "SZC": ("Công ty cổ phần Sonadezi Châu Đức", "Sonadezi Châu Đức"),
        "TCH": ("Công ty Cổ phần Đầu tư Dịch vụ Tài chính Hoàng Huy", "Tài chính Hoàng Huy"),
        "VCG": ("Tổng công ty cổ phần Xuất nhập khẩu và Xây dựng Việt Nam", "Vinaconex"),
        "VCI": ("Công ty cổ phần Chứng khoán Bản Việt", "Vietcap"),
        "VGC": ("Tổng công ty Viglacera", "Viglacera"),
        "VHC": ("Công ty cổ phần Vĩnh Hoàn", "Vĩnh Hoàn"),
        "VIX": ("Công ty cổ phần Chứng khoán VIX", "Chứng khoán VIX"),
        "VND": ("Công ty cổ phần Chứng khoán VNDIRECT", "VNDIRECT"),
        "VPI": ("Công ty cổ phần Đầu tư Văn Phú - Invest", "Văn Phú - Invest"),
        "VSH": ("Công ty cổ phần Thủy điện Vĩnh Sơn - Sông Hinh", "Thủy điện Vĩnh Sơn"),
        "TNH": ("Công ty Cổ phần Mới Hospital", "TNH Hospital"),
        "VOS": ("Công ty cổ phần Vận tải Biển Việt Nam", "Vosco"),
        "PET": ("Tổng Công ty Cổ phần Dịch vụ Tổng hợp Dầu khí", "Petrosetco"),
        "PVT": ("Tổng công ty cổ phần Vận tải Dầu khí", "PVTrans"),
        "DCM": ("Công ty cổ phần Phân bón Dầu khí Cà Mau", "Đạm Cà Mau"),
        "VPI": ("Công ty cổ phần Đầu tư Văn Phú - Invest", "Văn Phú Invest")
    }

    records = []
    # Sort for predictability
    for ticker in sorted(company_data.keys()):
        full_name, short_name = company_data[ticker]
        baseline_kw = f"{ticker}, {short_name}"
        records.append({
            'Ticker': ticker,
            'Tên_Công_ty': full_name,
            'Tên_Ngắn': short_name,
            'Baseline_Keywords': baseline_kw,
            'Custom_News_Keywords_User_Fill': baseline_kw  
        })

    # Convert to dataframe and deduplicate
    df = pd.DataFrame(records).drop_duplicates(subset=['Ticker'])
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"Đã cập nhật đúng toàn bộ bộ lọc VN100 + Viettel tại: {output_csv}")
    print(f"Tổng số mã thực tế: {len(df)}")

if __name__ == "__main__":
    fetch_data()
