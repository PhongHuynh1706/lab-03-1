# Lab 03 - Deep Q-Network

Repo này chứa tài liệu, notebook gốc và phần tách notebook ra file Python cho Lab 03.
Mục tiêu của phần tách `py/` là giúp code dễ đọc hơn trên Git, dễ kiểm kê theo từng
câu hỏi, và vẫn bám sát các block trong notebook để chép ngược lại khi cần.

## Cấu trúc thư mục

- `doc/`: đề bài và tài liệu Lab 03
- `notebooks/`: 2 notebook gốc được cung cấp cho lab
- `py/`: các block notebook được tách thành file Python và gom theo cụm câu hỏi
  - `q1/`: câu 1, gồm phần PyTorch basics và DQN trên FrozenLake
  - `q2_q3_vacuum/`: câu 2-3, gồm DQN trên VacuumCleanerEnv và phần ghi chú cải tiến
  - `q4_q5_q6_load_balancing/`: câu 4-6, gồm DQN trên LoadBalancingEnv và các scaffold cho metrics/reward redesign
  - `q7/`: câu 7, gồm flow Stable-Baselines3 cho VacuumCleanerEnv
  - `extras/`: các phần demo trong notebook không thuộc trực tiếp danh sách câu hỏi cuối PDF
- `requirements.txt`: danh sách thư viện cần thiết

## Quy ước hiện tại

- File được đánh số `00, 01, 02, ...` để dễ đọc theo thứ tự
- Đầu mỗi block có comment `# Source: ...` để map ngược về notebook/cell tương ứng
- Các file được giữ gần với nội dung notebook, hạn chế refactor mạnh để dễ copy-paste ngược
- Mỗi cụm runnable có một `main.py` chỉ làm orchestration, không chứa logic bài giải mới
- `q7/` hiện đang dùng lại `VacuumCleanerEnv` từ `q2_q3_vacuum/` cho tới khi thật sự cần một biến thể riêng cho SB3

## Chạy từng cụm câu hỏi

```bash
python py/q1/main.py
python py/q2_q3_vacuum/main.py
python py/q4_q5_q6_load_balancing/main.py
python py/q7/main.py
```

Các file `main.py` chỉ chạy tuần tự các block đã tách trong cùng một tiến trình Python,
để mô phỏng lại flow của notebook theo từng cụm câu hỏi.
