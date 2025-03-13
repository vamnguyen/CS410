# Differential Evolution (DE) và Cross-Entropy Method (CEM)

Dự án này cài đặt và so sánh hiệu năng của hai thuật toán tối ưu hóa: Differential Evolution (DE) và Cross-Entropy Method (CEM) trên các hàm mục tiêu khác nhau.

## Cài đặt

1. Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

## Cấu trúc dự án

- `src/`: Thư mục chứa mã nguồn
  - `objective_functions.py`: Định nghĩa các hàm mục tiêu
  - `de.py`: Cài đặt thuật toán Differential Evolution
  - `cem.py`: Cài đặt thuật toán Cross-Entropy Method
  - `visualization.py`: Các hàm vẽ đồ thị và tạo animation
  - `experiment.py`: Các hàm thực nghiệm
- `main.py`: File chính để chạy thực nghiệm
- `results/`: Thư mục chứa kết quả thực nghiệm
- `logs/`: Thư mục chứa log files
- `figures/`: Thư mục chứa đồ thị và animation

## Sử dụng

Để chạy thực nghiệm, sử dụng lệnh sau:

```bash
python main.py --mssv YOUR_MSSV
```

Trong đó `YOUR_MSSV` là mã số sinh viên của bạn, được sử dụng làm seed cho các lần chạy thực nghiệm.

## Các hàm mục tiêu

Dự án này sử dụng 5 hàm mục tiêu phổ biến:

1. Sphere
2. Griewank
3. Rosenbrock
4. Rastrigin
5. Ackley

Mỗi hàm được tối ưu hóa với 2 kích thước: d=2 và d=10.

## Kết quả

Sau khi chạy thực nghiệm, các kết quả sẽ được lưu trong các thư mục:

- `results/`: Chứa các bảng kết quả và file JSON chứa tất cả kết quả
- `logs/`: Chứa log files cho từng lần chạy
- `figures/`: Chứa đồ thị hội tụ và animation

## Tác giả

MSSV: 22520880
