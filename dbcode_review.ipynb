{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "gpuType": "T4",
      "authorship_tag": "ABX9TyMG/Au801CZsxUGAIVg3+6d",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/MrBIue/Le-Minh-Thuan-DBCDLTG-TTNTK46-4654100012/blob/main/dbcode.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "collapsed": true,
        "id": "PRMw4RQwOZ4Z"
      },
      "outputs": [],
      "source": [
        "# 1. Cài đặt thư viện\n",
        "!pip install neuralforecast pandas matplotlib\n",
        "\n",
        "# 2. Tải bộ dữ liệu ETTh1 chuẩn (7 biến)\n",
        "!wget -q -O ETTh1.csv https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv\n",
        "\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "# Đọc dữ liệu thô\n",
        "df_raw = pd.read_csv('ETTh1.csv')\n",
        "df_raw['date'] = pd.to_datetime(df_raw['date'])\n",
        "print(\"Đã tải dữ liệu thành công. Kích thước:\", df_raw.shape)\n",
        "df_raw.head()"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "# Tải data và chuẩn bị\n",
        "!wget -q -O ETTh1.csv https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv\n",
        "df_raw = pd.read_csv('ETTh1.csv')\n",
        "df_raw['date'] = pd.to_datetime(df_raw['date'])\n",
        "cols = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']\n",
        "\n",
        "# Vẽ bảng tổng hợp dữ liệu gốc\n",
        "fig, axes = plt.subplots(7, 1, figsize=(15, 20), sharex=True)\n",
        "for i, col in enumerate(cols):\n",
        "    axes[i].plot(df_raw['date'], df_raw[col], label=f'Gốc: {col}', color='#2a9d8f')\n",
        "    axes[i].set_ylabel(col)\n",
        "    axes[i].legend(loc='upper right')\n",
        "plt.suptitle('Tổng quan 7 biến số trong bộ dữ liệu ETTh1 (Toàn bộ thời gian)', fontsize=16)\n",
        "plt.show()"
      ],
      "metadata": {
        "id": "4fS0rgoOOfYL"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#CELL LỖI (KHÔNG CẦN)\n",
        "#from neuralforecast import NeuralForecast\n",
        "#from neuralforecast.models import PatchTST\n",
        "#from neuralforecast.losses.pytorch import MSE\n",
        "\n",
        "# Chuyển sang định dạng Long Format\n",
        "#df_nf = df_raw.melt(id_vars=['date'], value_vars=cols, var_name='unique_id', value_name='y')\n",
        "#df_nf.rename(columns={'date': 'ds'}, inplace=True)\n",
        "#df_nf = df_nf.sort_values(['unique_id', 'ds']).reset_index(drop=True)\n",
        "\n",
        "# Chia tỷ lệ 7:1:2\n",
        "#n_total = len(df_raw)\n",
        "#train_end = int(n_total * 0.7)\n",
        "#val_end = int(n_total * 0.8)\n",
        "\n",
        "#train_df = df_nf.groupby('unique_id').head(train_end)\n",
        "#val_df = df_nf.groupby('unique_id').apply(lambda x: x.iloc[train_end:val_end]).reset_index(drop=True)\n",
        "#test_df = df_nf.groupby('unique_id').tail(n_total - val_end)\n",
        "\n",
        "#print(f\"Hoàn tất chia dữ liệu. Train: {train_end}, Val: {val_end-train_end}, Test: {n_total-val_end}\")"
      ],
      "metadata": {
        "id": "xKw43KsqOgp3",
        "collapsed": true
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from neuralforecast import NeuralForecast\n",
        "from neuralforecast.models import PatchTST\n",
        "from neuralforecast.losses.pytorch import MSE\n",
        "\n",
        "# 1. Tiền xử lý chọn riêng LUFL\n",
        "target = 'LUFL'\n",
        "df_target = df_raw[['date', target]].copy()\n",
        "df_target.columns = ['ds', 'y']\n",
        "df_target['unique_id'] = target\n",
        "\n",
        "# 2. Chia tỷ lệ 7:1:2\n",
        "n = len(df_target)\n",
        "train_df = df_target.iloc[:int(n*0.7)]\n",
        "val_df = df_target.iloc[int(n*0.7):int(n*0.8)]\n",
        "test_df = df_target.iloc[int(n*0.8):]\n",
        "\n",
        "# 3. Cấu hình PatchTST Nâng cấp (Tăng khả năng học)\n",
        "model_lufl = PatchTST(\n",
        "    h=96,\n",
        "    input_size=512,        # Nhìn lại dài hơn (336 -> 512)\n",
        "    patch_len=16,\n",
        "    stride=8,\n",
        "    hidden_size=256,       # Mạng rộng hơn (128 -> 256)\n",
        "    n_heads=8,             # Attention mạnh hơn (4 -> 8)\n",
        "    scaler_type='standard',\n",
        "    max_steps=2000,        # Học kỹ hơn (1500 -> 2000)\n",
        "    learning_rate=1e-4     # Tốc độ học nhỏ để hội tụ chính xác\n",
        ")\n",
        "\n",
        "nf = NeuralForecast(models=[model_lufl], freq='H')\n",
        "nf.fit(df=pd.concat([train_df, val_df]), val_size=len(val_df))\n",
        "\n",
        "# 4. Dự báo và Xuất biểu đồ \"vàng\" cho tiểu luận\n",
        "forecasts = nf.predict().reset_index()\n",
        "actual = test_df.head(96)\n",
        "\n",
        "plt.figure(figsize=(15, 6))\n",
        "plt.plot(actual['ds'], actual['y'], label='Thực tế (LUFL)', color='#1f77b4', marker='o', markersize=4)\n",
        "plt.plot(actual['ds'], forecasts['PatchTST'], label='Dự báo Nâng cấp (PatchTST)', color='#e76f51', linestyle='--', linewidth=2.5)\n",
        "plt.title(f'KẾT QUẢ DỰ BÁO TỐI ƯU CHO BIẾN {target}', fontsize=14, fontweight='bold')\n",
        "plt.legend()\n",
        "plt.grid(True, alpha=0.3)\n",
        "plt.show()"
      ],
      "metadata": {
        "id": "ACXCC_dAOwj5"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.metrics import mean_squared_error, mean_absolute_error\n",
        "import numpy as np\n",
        "\n",
        "# --- 1. TÍNH TOÁN SAI SỐ (METRICS) ---\n",
        "# Lấy kết quả dự báo từ biến 'forecasts' và thực tế từ 'actual' đã có ở Cell 5\n",
        "y_true = actual['y'].values\n",
        "y_pred = forecasts['PatchTST'].values\n",
        "\n",
        "mse = mean_squared_error(y_true, y_pred)\n",
        "mae = mean_absolute_error(y_true, y_pred)\n",
        "rmse = np.sqrt(mse)\n",
        "\n",
        "print(f\"--- KẾT QUẢ ĐỊNH LƯỢNG (Biến LUFL) ---\")\n",
        "print(f\"MSE (Sai số bình phương trung bình): {mse:.4f}\")\n",
        "print(f\"MAE (Sai số tuyệt đối trung bình): {mae:.4f}\")\n",
        "print(f\"RMSE (Căn bậc 2 của MSE): {rmse:.4f}\")\n",
        "\n",
        "# --- 2. LƯU MÔ HÌNH (MODEL PERSISTENCE) ---\n",
        "# Lưu lại để lần sau chỉ cần Load, không cần train lại tốn tài nguyên T4\n",
        "nf.save(path='./patchtst_lufl_model', overwrite=True)\n",
        "print(\"\\n--- ĐÃ LƯU MÔ HÌNH THÀNH CÔNG ---\")\n",
        "print(\"Bạn có thể tải thư mục 'patchtst_lufl_model' về máy tính để dự phòng.\")"
      ],
      "metadata": {
        "id": "wasNRk45ZbZ4"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}
