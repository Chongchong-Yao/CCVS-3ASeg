from PIL import Image
import base64
import os


def jpg_to_svg_embedded(jpg_path, svg_path):
    """
    将单个JPG图片转换为SVG格式（嵌入base64）

    Args:
        jpg_path (str): JPG图片路径
        svg_path (str): 输出SVG文件路径
    """
    try:
        with Image.open(jpg_path) as img:
            # 确保图片是RGB模式（JPG不支持透明度）
            if img.mode in ('RGBA', 'P', 'LA'):
                img = img.convert('RGB')

            width, height = img.size

            # 将图片转换为base64编码
            def image_to_base64(image):
                from io import BytesIO
                buffer = BytesIO()
                # 使用JPEG格式保存，质量设为85以减小文件大小
                image.save(buffer, format="JPEG", quality=85)
                return base64.b64encode(buffer.getvalue()).decode('ascii')

            img_base64 = image_to_base64(img)

            # 创建SVG内容
            svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <image width="{width}" height="{height}" xlink:href="data:image/jpeg;base64,{img_base64}"/>
</svg>'''

            # 确保输出目录存在
            os.makedirs(os.path.dirname(svg_path), exist_ok=True)

            # 保存SVG文件
            with open(svg_path, 'w', encoding='utf-8') as f:
                f.write(svg_content)

            print(f"转换成功: {os.path.basename(jpg_path)} → {os.path.basename(svg_path)} ({width}x{height})")

            # 检查生成的SVG文件大小
            svg_size = os.path.getsize(svg_path)
            print(f"  SVG文件大小: {svg_size} 字节")

    except Exception as e:
        print(f"转换失败 {os.path.basename(jpg_path)}: {e}")


def batch_convert_jpg_to_svg(input_folder, output_folder=None):
    """
    批量转换文件夹中的所有JPG文件为SVG

    Args:
        input_folder (str): 包含JPG文件的输入文件夹路径
        output_folder (str): 输出SVG文件的文件夹路径，如果为None则使用输入文件夹
    """
    # 检查输入文件夹是否存在
    if not os.path.isdir(input_folder):
        print(f"错误: 输入文件夹 '{input_folder}' 不存在")
        return

    # 设置输出文件夹
    if output_folder is None:
        output_folder = input_folder
    else:
        # 创建输出文件夹（如果不存在）
        os.makedirs(output_folder, exist_ok=True)
        print(f"输出文件夹: {output_folder}")

    # 统计信息
    converted_count = 0
    skipped_count = 0

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 检查是否为JPG文件（不区分大小写）
        if filename.lower().endswith(('.jpg', '.jpeg')):
            jpg_path = os.path.join(input_folder, filename)

            # 生成对应的SVG文件名（替换扩展名）
            svg_filename = os.path.splitext(filename)[0] + '.svg'
            svg_path = os.path.join(output_folder, svg_filename)

            # 避免重复转换（如果SVG已存在则跳过）
            if os.path.exists(svg_path):
                print(f"已存在，跳过: {svg_filename}")
                skipped_count += 1
                continue

            # 转换单个文件
            jpg_to_svg_embedded(jpg_path, svg_path)
            converted_count += 1

    print(f"\n批量转换完成!")
    print(f"转换文件数: {converted_count}")
    print(f"跳过文件数: {skipped_count}")
    print(f"输出位置: {output_folder}")


def batch_convert_with_subfolders(input_folder, output_base_folder=None):
    """
    批量转换文件夹及其子文件夹中的所有JPG文件为SVG，保持目录结构

    Args:
        input_folder (str): 包含JPG文件的输入文件夹路径
        output_base_folder (str): 输出SVG文件的基础文件夹路径
    """
    if output_base_folder is None:
        output_base_folder = input_folder + "_svg"

    # 创建输出基础文件夹
    os.makedirs(output_base_folder, exist_ok=True)

    converted_total = 0

    for root, dirs, files in os.walk(input_folder):
        # 计算相对路径，用于在输出文件夹中创建相同的目录结构
        relative_path = os.path.relpath(root, input_folder)
        if relative_path == '.':
            output_folder = output_base_folder
        else:
            output_folder = os.path.join(output_base_folder, relative_path)

        # 创建输出子文件夹
        os.makedirs(output_folder, exist_ok=True)

        # 统计当前文件夹的转换数量
        folder_converted = 0

        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg')):
                jpg_path = os.path.join(root, filename)
                svg_filename = os.path.splitext(filename)[0] + '.svg'
                svg_path = os.path.join(output_folder, svg_filename)

                # 避免重复转换
                if not os.path.exists(svg_path):
                    jpg_to_svg_embedded(jpg_path, svg_path)
                    folder_converted += 1
                    converted_total += 1

        if folder_converted > 0:
            print(f"文件夹 '{relative_path}' 转换了 {folder_converted} 个文件")

    print(f"\n全部转换完成! 总共转换了 {converted_total} 个文件")
    print(f"输出位置: {output_base_folder}")


# 使用示例
if __name__ == "__main__":
    # 输入文件夹路径（包含JPG文件）
    input_folder = r"D:\myDataManager\pycharmProject\Crack-Segmentation\FGEM\crack\test_result\predict_images\select"

    # 方式1: 简单输出到指定文件夹
    output_folder = r"D:\myDataManager\pycharmProject\Crack-Segmentation\FGEM\crack\test_result\predict_images\select\svg_out"
    batch_convert_jpg_to_svg(input_folder, output_folder)

    print("\n" + "=" * 50 + "\n")
