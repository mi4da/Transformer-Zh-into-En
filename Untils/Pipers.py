import os
import time
class PIPERS:
    def __init__(self):
        self.Index = {
            # 阿里
            "https://mirrors.aliyun.com/pypi/simple/",
            # 清华
            "https://pypi.tuna.tsinghua.edu.cn/simple/",
            # 中科大
            "https://mirrors.bfsu.edu.cn/pypi/web/simple/",
            # 清华
            "https://pypi.tuna.tsinghua.edu.cn/simple/",
            # douban
            "http://pypi.doubanio.com/simple"
        }
        self.libs = {"numpy", "matplotlib", "pillow", "gooey", "sklearn",
                     "requests", "pyinstaller", "django", "jieba",
                     "beautifulsoup4", "wheel", "networkx", "sympy",
                     "flask", "werobot", "pyqt5", "pandas", "pyopengl",
                     "pypdf2", "docopt", "pygame", "peewee", "scrapy",
                     "behold", "cython"}
    def __printlog(self):
        print("如果运行失败，请使用管理员权限启动python进程！")
        time.sleep(5)
    def ChageResponsities(self,IndexUrls = None):
        self.__printlog()
        if IndexUrls is None:
            self.UserIndex = self.Index
        else:
            self.UserIndex = IndexUrls
        try:
            for i in self.UserIndex:
                os.system("pip config set global.index-url " + i)
            print("添加源成功！")
        except:
            print("添加源失败！")
    def InstallCommonTools(self,UserLibs = None):
        self.__printlog()
        if UserLibs is None:
            self.UserLibs = self.libs
        else:
            self.UserLibs = UserLibs
            # for lib in self.UserLibs:
            #     os.system("pip install " + lib)
            # print("常用库成功安装！")

            try:
                for lib in self.UserLibs:
                    os.system("pip install --user " + lib)
                print("常用库成功安装！")
            except:
                print("常用库安装失败！")



