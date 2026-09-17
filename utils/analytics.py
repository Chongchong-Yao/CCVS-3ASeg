import torch

def main2():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 模型初始化
    model = EEM_Laplace(ch_in=3, ch_out=3).to(device)
    model.eval()

    # === 瓶颈分析 ===
    print("\n" + "=" * 50)
    print("模块瓶颈分析")
    print("=" * 50)

    # 定义输入形状 (batch_size, channels, height, width)
    input_shape = (8, 3, 640, 640)


    # 打印结果
    print(f"{'操作步骤':<20} {'时间(ms)':<12} {'占比(%)':<10}")
    print("-" * 50)
    total_time = 0
    for op_name, timing_info in timings.items():
        time_ms = timing_info['time_ms']
        percentage = timing_info['percentage']
        total_time += time_ms
        print(f"{op_name:<20} {time_ms:<12.3f} {percentage:<10.1f}")

    print(f"{'总计':<20} {total_time:<12.3f} {100:<10.1f}")

    # 识别性能瓶颈
    bottleneck = max(timings.items(), key=lambda x: x[1]['percentage'])
    print(f"\n主要性能瓶颈: {bottleneck[0]} ({bottleneck[1]['percentage']:.1f}%)")


if __name__ == "__main__":
    main2()