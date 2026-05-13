# NCKH Abstract and Keywords

## Vietnamese Title

Đánh giá thực nghiệm các mô hình học máy và ensemble trong dự báo xu hướng cổ phiếu VN100 theo phương pháp walk-forward ngoài mẫu.

## English Title

Walk-Forward Evaluation of Machine Learning and Ensemble Models for VN100 Stock Direction Forecasting.

## Vietnamese Abstract

Nghiên cứu này đánh giá các mô hình học máy và ensemble trong dự báo xu hướng cổ phiếu VN100 theo thiết kế walk-forward có kiểm soát rò rỉ dữ liệu. Bộ bằng chứng sử dụng các artifact chính thức tại `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff`, trong đó nhãn huấn luyện được giới hạn đến ngày 2024-12-31 và kết quả năm 2025 được dùng làm giai đoạn đánh giá ngoài mẫu. Các mô hình được xem xét gồm LightGBM, XGBoost, random forest và stacking, so sánh với các baseline định hướng đơn giản. Kết quả benchmark chính thức cho thấy độ chính xác định hướng tổng thể của dữ liệu daily là 53,19% trên 26.104 dự báo và của dữ liệu hourly là 51,29% trên 127.944 dự báo; cả hai đều không vượt ngưỡng benchmark toàn cục 60%. Tuy nhiên, các chẩn đoán có điều kiện cho thấy tín hiệu dự báo trong một số lát cắt: hourly stacking với horizon 1 và ngưỡng confidence 0,57 đạt 60,03% độ chính xác với 31,30% coverage, trong khi một số chẩn đoán bear-regime daily horizon 20 vượt 63% độ chính xác. Các kết quả này bị giới hạn bởi coverage cache còn hẹp, chỉ bảy mã được đánh giá, bằng chứng confidence sweep chưa đầy đủ, mức tập trung ticker trong lát cắt được chọn và thiếu artifact về chi phí giao dịch, trượt giá, turnover, drawdown, profit factor và lợi nhuận sau chi phí. Vì vậy, nghiên cứu đóng góp một khung benchmark VN100 có khả năng tái lập và ranh giới diễn giải thận trọng: có tín hiệu dự báo có điều kiện, nhưng chưa có bằng chứng về việc vượt benchmark toàn cục, đại diện toàn bộ VN100, phương pháp ổn định 63% cho toàn thị trường hoặc sẵn sàng giao dịch thực tế.

## English Abstract

This study evaluates machine learning and ensemble models for VN100 stock-direction forecasting under a leakage-aware walk-forward design. The evidence base uses the official artifact family `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff`, which applies a 2024-12-31 training-label cutoff and evaluates 2025 outcomes out of sample. The evaluated models are LightGBM, XGBoost, random forest, and stacking, compared with simple directional baselines. The official benchmark records 53.19% daily directional accuracy over 26,104 predictions and 51.29% hourly directional accuracy over 127,944 predictions; neither frequency passes the global 60% benchmark threshold. Conditional diagnostics nevertheless show selected predictive signal: hourly stacking with horizon 1 and confidence threshold 0.57 reaches 60.03% filtered accuracy at 31.30% coverage, while selected daily bear-regime horizon-20 diagnostics exceed 63% accuracy. These results are constrained by limited benchmark-usable cache coverage, seven evaluated tickers, incomplete confidence-sweep evidence, concentration in the selected confidence slice, and the absence of official transaction-cost, slippage, turnover, drawdown, profit-factor, and cost-adjusted return artifacts. The study contributes a reproducible VN100 benchmark framework and a disciplined claim boundary: conditional predictive signal is present in the official artifacts, but global benchmark success, full-market representativeness, stable 63% performance, and practical trading readiness are not established.

## Vietnamese Keywords

VN100; dự báo xu hướng cổ phiếu; walk-forward; học máy; ensemble; stacking; độ chính xác định hướng; lọc confidence; chẩn đoán regime; kiểm soát rò rỉ dữ liệu.

## English Keywords

VN100; stock direction forecasting; walk-forward validation; machine learning; ensemble; stacking; directional accuracy; confidence filtering; regime diagnostics; leakage control.
