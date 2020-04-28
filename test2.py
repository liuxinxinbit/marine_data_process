import os, re, time


name = 'json'
flags = True

while flags:
    for f in os.listdir('/home/liuxinxin/marine_data/01/images'):
　　　　　# 寻找以name开头，以.download结尾的文件
        if re.search(name + '\.(.*)\.download$', f):
            print f
            print '没有完成'
            time.sleep(5)
    #flags =False
print '完成'
# 当前时间
print time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())



