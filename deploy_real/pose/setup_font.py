"""
下载并配置中文字体用于matplotlib
"""
import os
import urllib.request
import matplotlib.font_manager as fm

# 获取matplotlib字体目录
mpl_data_dir = os.path.dirname(fm.matplotlib_fname())
font_dir = os.path.join(mpl_data_dir, 'fonts', 'ttf')

# 下载思源黑体（开源中文字体）
font_url = "https://github.com/adobe-fonts/source-han-sans/raw/release/OTF/SimplifiedChinese/SourceHanSansSC-Regular.otf"
font_path = os.path.join(font_dir, "SourceHanSansSC-Regular.otf")

if not os.path.exists(font_path):
    print("正在下载中文字体...")
    try:
        urllib.request.urlretrieve(font_url, font_path)
        print(f"字体已下载到: {font_path}")
        
        # 清理matplotlib缓存
        cache_dir = os.path.join(os.path.expanduser("~"), '.matplotlib')
        cache_file = os.path.join(cache_dir, 'fontlist-v330.json')
        if os.path.exists(cache_file):
            os.remove(cache_file)
            print("已清理字体缓存")
        
        # 重建字体管理器
        fm._load_fontmanager(try_read_cache=False)
        print("字体配置完成！")
        
    except Exception as e:
        print(f"字体下载失败: {e}")
        print("请手动安装中文字体或使用英文标签")
else:
    print(f"字体已存在: {font_path}")
