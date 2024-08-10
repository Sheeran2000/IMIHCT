# from PIL import Image
#
# # 打开图片和mask
# image = Image.open(r'C:\Users\xingfang\Desktop\微信图片_20240424094940.jpg')
# mask = Image.open(r'E:\image_inpainting\Datasets\test_mask_256\mask30-40\06000.png')
#
# # 将mask覆盖在图片上
# image.paste(mask, (0, 0), mask)
# image = image.convert('RGB')
# # 保存图片
# image.save('7.jpg')

from PIL import Image


def resize_image(image_path, output_path, size=(256, 256)):
    """
    将图像调整为指定大小，并保存到输出路径。

    参数：
    image_path: 输入图像的路径。
    output_path: 调整大小后的图像保存路径。
    size: 要调整的目标大小，默认为 (256, 256)。
    """
    try:
        # 打开图像文件
        with Image.open(image_path) as img:
            # 调整图像大小
            resized_img = img.resize(size)
            # 保存调整大小后的图像
            resized_img.save(output_path)
            print("图像已成功调整大小并保存到:", output_path)
    except Exception as e:
        print("发生错误:", e)


# 调用函数来调整图像大小
input_image_path = r'C:\Users\xingfang\Desktop\微信图片_20240424094940.jpg'  # 输入图像路径
output_image_path = r'C:\Users\xingfang\Desktop\压缩256.jpg'  # 输出图像路径
resize_image(input_image_path, output_image_path)
