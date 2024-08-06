from logger import Logger
import time

# 'data.slen_10.tlen_1.seq.patlen_1.lit.patlen_8.nitems_5000_spmf.txt' (dataset1)
# 'data.slen_10.tlen_1.seq.patlen_2.lit.patlen_8.nitems_5000_spmf.txt' (dataset2)
# 'test.txt'
# 'sign language.txt'
filename = ('./datasets/test.txt')
saveDir = './outputs/'
min_support = 2
log_file_name = (saveDir + "log-" + time.strftime("%Y%m%d-%H%M%S",
                                                  time.localtime()) + f"-minSupport-{min_support}" + ".log")
log = Logger(log_file_name)
count = 0


# 读数据集
def readfile(name):
    S, union = [], []
    with open(name, 'r') as input1:
        for line in input1.readlines():
            tmp = []
            elements = line.split(' -1 ')[:-1]
            for i in elements:
                items = i.split(' ')
                tmp.append(items)
                union += items
            S.append(tmp)
    union_item = list(set(union))
    return S, len(S), union_item


# 得到后缀, ll为序列，da为前缀
def dd(ll, da):
    flag = 0
    str1 = []
    for l in ll:
        if flag == 0:
            if len(l) == 1 and l[0] == da:
                flag = 1
                continue
            elif len(l) > 1:
                if l[-1] == da:
                    flag = 1
                    continue
                tmp = []
                for i in l:
                    if flag == 0:
                        tmp = []
                        if i == da:
                            flag = 1
                            tmp.append('_')
                    else:
                        tmp.append(i)
                if len(tmp) > 0:
                    str1.append(tmp)
        else:
            str1.append(l)
    # 返回找到的后缀, 如未找到str1为空
    return str1


# 前缀与后缀
def first_step(un, S, support, new_list=None):
    global count
    list2 = []
    dict1 = {}
    if new_list is None:
        for lab in un:  # 遍历第一层序列的前缀列表
            list_all = []
            for item in S:  # 遍历第一层序列 找后缀
                alter = dd(item, lab)
                if len(alter) > 0:
                    list_all.append(alter)
            if len(list_all) >= support:
                log.info(f'以 {lab} 为前缀, 支持度: {len(list_all)}, 后缀为:')
                count += 1
                for i in list_all:
                    log.info(i)
                list1 = [tuple(tuple([y for y in x]) for x in [[lab]]), list_all]  # 单个前缀和其所有后缀
                list2.extend(list1)  # 前缀和其所有后缀集
                # 键和值交替出现
                dict1 = dict(zip(list2[::2], list2[1::2]))  # 将前缀作为键，后缀作为值，创建字典
    else:
        for lab1 in new_list:
            list_all = []
            for item in S:  # 遍历第一层序列，调用dd
                alter = dd(item, lab1[-1][-1])
                if len(alter) > 0:
                    list_all.append(alter)
            if len(list_all) >= support:
                log.info(f'以 {lab1} 为前缀, 支持度: {len(list_all)}, 后缀为:')
                count += 1
                for i in list_all:
                    log.info(i)
                list1 = [tuple(tuple([y for y in x]) for x in lab1), list_all]
                list2.extend(list1)
                dict1 = dict(zip(list2[::2], list2[1::2]))  # 将前缀作为键，后缀作为值，创建字典

    return dict1

# 找到大于min_support的前缀
def match_dict(x, min_support):
    list11 = []
    for key in x.keys():
        if x[key] >= min_support:
            list11.append(key)
    return list11


# 获取某前缀下后缀的集合, 计算各个元素支持度
def mm(list_all, min_support):
    listX, list_ = [], []
    for items in list_all:
        for item in items:
            flag = 0
            for i in item:
                if flag == 0:
                    if i == '_':  # 以_开头, 与原key在一个子序列中
                        flag = 1
                    else:
                        listX.append(i)
                else:
                    list_.append('_' + i)
                    flag = 0
    set_ = set(list_)
    dict_ = {}
    for item in set_:
        dict_.update({item: list_.count(item)})  # 计算后缀出现次数形成字典
    list_y = match_dict(dict_, min_support)  # 二阶前缀（含_），支持度大于 min support

    setX = set(listX)
    dictX = {}
    for item in setX:
        dictX.update({item: listX.count(item)})  # 计算后缀出现次数形成字典
    list_out = match_dict(dictX, min_support)  # 二阶前缀，支持度大于 min support
    if len(list_y) > 0:  # 维护包含'_'的前缀
        list_out.extend(list_y)
    return list_out  # 返回下一阶前缀


def next_step(dict2, min_support, step):
    for key, value in dict2.items():
        key = list(list(items) for items in list(key))
        list_x = mm(value, min_support)
        new_list = list_x.copy()
        new_key = []
        for j in new_list:
            old_key = key.copy()
            if j.startswith("_"):
                old_key[-1].append(j[1:])
                new_key.append(old_key)
            else:
                old_key.append([j])
                new_key.append(old_key)
        if len(list_x) > 0:
            dict1_2 = first_step(list_x, value, min_support, new_key)
            if len(dict1_2) > 1:
                log.info(f"递归处理第 {step} 阶：")
                next_step(dict1_2, min_support, step + 1)


if __name__ == '__main__':
    T1 = time.process_time()
    S, support, un = readfile(filename)
    log.info(f'数据集: {filename}')
    log.info(f'设最小支持度为 {min_support}\n')
    dict1 = first_step(un, S, min_support, None)  # 前缀和后缀
    T2 = time.process_time()
    log.info(f'第一轮挖掘结束, 耗时 {T2 - T1}')
    log.info("递归处理第 2 阶：")
    next_step(dict1, min_support, 3)  # 递归
    T3 = time.process_time()
    log.info(f'\n\n\n挖掘结束, 耗时 {T3 - T1} s\n共挖掘到 {count} 个频繁序列')
