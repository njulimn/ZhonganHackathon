'''
@author: 程潇
@team: 生男孩48
'''

import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import datetime, random, os

'''
导入csv文件数据
'''
file_names = ['policy', 'customer', 'claim', 'renewal']
Df = {}
header = {}
for fn in file_names:
    #csv_path = 'data_new/' + fn + '_sample_new.csv'
    csv_path = 'data/' + fn + ('_train' if fn == 'renewal' else '') + '.csv'
    print(fn, ':', csv_path)
    Df[fn] = pd.read_csv(csv_path)
    Df[fn] = Df[fn].sort_values(by=['policy_id']).reset_index(drop=True)
    print('  shape:', Df[fn].shape)
    hd = list(Df[fn])
    header[fn] = {}
    for k, group in (itertools.groupby(hd, key=lambda x:x[0])):
        header[fn][k] = len(list(group))
    print(' ', header[fn], '\n')

# 统计各表中的数据    
for fn in file_names:
    print('\n', fn, Df[fn].shape)
    for k in "pcvzl":
        if k not in header[fn]: continue
        for i in range(header[fn][k]):
            key = '%s%02d'%(k,i)
            if k=='p': key = 'policy_id'
            elif k=='l': key = 'label'
            column = Df[fn][key]
            t = column.value_counts().shape[0]
            print('  {}: {:5d}\t{:.2f}'.format(key, t, t/Df[fn].shape[0]), end= " ")
            #if t/Df[fn].shape[0]<0.4:
                #print(column.value_counts())
            if k=='v' and fn not in ['customer']: 
                print('Max: {:.2f}, Min: {:.2f}, Mean: {:.2f}'.format(column.max(), column.min(), column.mean()))
            else: print()
labels = df['renewal']['label']


#############################################################
'''
定义各种数据的对象及数据处理方法
'''
NAN = -999  # 默认缺省数据填充值
def Func(v, h):
    if pd.isnull(v) or x=='\\N':
        return 0 if h[0]=='v' else NAN
    else: return v

class Policy:
    '''
    Policy 类，用于统计和整理 policy.csv 的数据
    '''
    key = 'policy'
    remove = {'c03', 'c04', 'c07', 'c10', 'z08', 'z09', 'z10'}
    remove = {}
    potiential_remove = {'c05', 'c11', 'z01', 'z03', 'z04', 'z05'}
    potiential_remove = {}
    province = ['四川省', '湖北省', '广东省', '江苏省', '山东省', '河南省', '浙江省', '湖南省', '上海市', '河北省',
       '福建省', '安徽省', '黑龙江省', '江西省', '上海', '辽宁省', '广西壮族自治区', '吉林省', '北京', '陕西省',
       '山西省', '内蒙古自治区', '云南省', '重庆', '贵州省', '天津', '新疆维吾尔族自治区', '甘肃省', '海南省',
       '宁夏回族自治区', '北京市', '青海省', '天津市', '重庆市', '杭州', '西藏自治区']
    insure_price = sorted([0, 1000000, 3000000, 300000, 500000])
    head = []
    type = []
    dict_id = {} # 存放各 policy id 对应的数据项（policy 对象）
    def __init__(self):
        self.id = None
        self.data = {}
        self.series = []
    def process(self, series):
        self.series = series
        self.id = self.series['policy_id']
        for head in self.head:
            if head=='z00':
                self.data[head] = Policy.province.index(series[head])
            elif head=='v00':
                self.data[head] = Policy.insure_price.index(series[head])
            else:
                self.data[head] = Func(self.series[head], head)
        
    def data_process():
        print('Policy Processing ...')
        df = Df[Policy.key]
        #province = df['z00'].tolist.sort()
        
        for index, row in df.iterrows():
            obj = Policy()
            obj.process(row)
            Policy.dict_id[obj.id] = obj
        print('done.')
    def get(self, hds):
        return [self.data[hd] for hd in hds]
        
class Customer:
    '''
    Customer 类，用于统计和整理 customer.csv 的数据, 并建立与 policy 的索引关系
    '''
    key = 'customer'
    remove = {'z02', 'v16', 'v00', 'v11', 'v13', 'v16'}
    remove = {}
    potiential_remove = {'v02', 'v03', 'v04', 'v05', 'v06', 'v07', 'v08' 'v09', 'v15'}
    potiential_remove = {}
    head = []
    type = []
    dict_id = {} #存放 policy id 对应的customer数据项
    dict_user = {} # 存放用户对应的 policy id 数据
    avg_birth = 0
    def __init__(self):
        self.id = None
        #self.holder_data = []
        #self.insured_data = []
        self.data = {}
        self.series = []
        
    def process(self, series):
        if pd.isnull(series['c08']): series['c08'] = '1'
        series['c08'] = '0' if series['c08'] in 'F0' else '1'
        self.series = (series)
        self.id = series['policy_id']
        temp = {}
        for head in Customer.head:
            v = series[head]
            if head == 'z03':
                #print(v)
                if pd.isnull(v): temp[head] = (Customer.avg_birth)
                #if pd.isnull(v): temp[head] = NAN
                else: 
                    #print(v, 2018-int(v[:v.find('-')]))
                    temp[head] = (2018-int(v[:v.find('-')]))
            else:
                temp[head] = (Func(v, head))
        self.data = temp
        #user = "{}{}".format(series['c08'], series['z03']) #gender+birthday
        user = series['c02']
        if user in Customer.dict_user:
            if self.id not in Customer.dict_user[user]:
                Customer.dict_user[user].append(self.id)
        else:
            Customer.dict_user[user] = [self.id]
    def get(self, hds):
        return [self.data[hd] for hd in hds]
        
    def data_process():
        print('Customer Processing ...')
        df = Df[Customer.key]
        cnt, s = 0.0, 0.0
        for y in df['z03']:
            if not pd.isnull(y):
                s += (2018-float(y[:y.find('-')]))
                cnt += 1
        Customer.avg_birth = s/cnt
            
        #cnt = 0
        for index, row in df.iterrows():
            #cnt += 1
            #if cnt>2000: break
            obj = Customer()
            obj.process(row)
            if obj.id not in Customer.dict_id:
                Customer.dict_id[obj.id] = [obj]
            else: Customer.dict_id[obj.id].append(obj)
        print('done.')
        
        
class Claim:
    '''
    Claim 类，用于统计和整理 claim.csv 的数据, 并建立与 policy id 的索引关系
    '''
    key = 'claim'
    accept = ["正常赔付", "协议赔付", "通融赔付"]
    refuse = ["拒赔","报案注销", "报案回退"]
    remove = {'c00', 'c01', 'c07', 'z01','z02','z03', 'z05'}
    remove = {}
    potiential_remove = {}#{'c02', 'c03', 'c04', 'c05', 'c06'}
    head = []
    type = []
    dict_id = {} # 存放 policy id 对应的 claim 数据
    def __init__(self):
        self.id = None
        self.data = {}
        self.series = []
    def process(self, series):
        label = series['z06']
        if label in self.accept:
            series['z06'] = '1'
        elif label in self.refuse:
            series['z06'] = '-1'
        else:
            series['z06'] = '0'
        #for v in ['v00', 'v01', 'v02']:
            #if pd.isnull(series[v]): series[v] = 0
        self.series.append(series)
        self.id = series['policy_id']
        self.data = {head: Func(series[head], head) for head in Claim.head}
    def get(self, hds):
        return [self.data[hd] for hd in hds]
    def data_process():
        print("Claim Processing ...")
        df = Df[Claim.key]
        for index, row in df.iterrows():
            obj = Claim()
            obj.process(row)
            if obj.id in Claim.dict_id:
                Claim.dict_id[obj.id].append(obj)
            else: Claim.dict_id[obj.id] = [obj]
        print('done.')
        
def pre_process():
    '''
    初步处理一些头部数据
    '''
    print('pre processing ...')
    for _, cla in Class.items():
        for tp in 'cvz':
            for i in range(header[cla.key][tp]):
                head = '%s%02d'%(tp,i)
                if head not in cla.remove and head not in cla.potiential_remove:
                    cla.head.append(head)
                    cla.type.append('v' if tp=='v' else 'c')
    print('done.')
                    
def export_csv(key):
    print('Export data to', key+'_processed.csv')
    table = Class[key]
    with open(key+'_processed.csv', 'w') as file:
        file.write('policy_id,' + ','.join(table.head) + '\n')
        if key is 'policy':
            for pid, obj in table.dict_id.items():
                file.write(str(pid) + ','.join(map(str, obj.data)) + '\n')
        else:
            for pid, obj in table.dict_id.items():
                for i in range(len(obj.data)):
                    file.write(str(pid) + ','.join(map(str, obj.data[i])) + '\n')
            

Class = {'policy': Policy, 'customer': Customer, 'claim': Claim}

# 预处理
pre_process()

# 处理 policy 数据
Policy.data_process()
# 处理 customer 数据
Customer.data_process()
# 处理 claim 数据
Claim.data_process()
# 处理 renewal 数据
Renewal = {}
for _, row in Df['renewal'].iterrows():
    Renewal[row['policy_id']] = row['label']
print(len(Renewal.items()))

#########################################################
'''
提取并整合数据特征
'''

Claim_feature = {}
for pid, objs in Claim.dict_id.items():
    count = [0,0,0]
    for k, g in itertools.groupby(objs, key=lambda obj:obj.data['z06']):
        count[int(k)+1] = len(set(g)) 
    s0, s1 = 0, 0
    for obj in objs:
        if obj.data['v01'] != NAN:
            s1 += obj.data['v01']
        if obj.data['v00'] != NAN:
            s0 += obj.data['v00']
    Claim_feature[pid] = [100*s0/(1+Policy.dict_id[pid].get(['v00'])[0]), 100*s1/(1+Policy.dict_id[pid].get(['v01'])[0])]
    Claim_feature[pid].extend(count)



time = datetime.datetime.now()
print(time, 'saving data ...')
test_file_path = 'ftp/test/policy_customer_test {}.csv'.format(NAN, time)
#test_file_path = 'ftp/test/test.csv'
features_path = 'ftp/feature/feature {}.txt'.format(time)
train_file_path = 'ftp/policy_customer_train {}.csv'.format(NAN, time)
#train_file_path = 'ftp/data.csv'
test_data = {}
train_data = {}
features = []
count = [0, 0]
count_add = [0, 0]


# 整合数据特征并保存到文件
with open(train_file_path, 'w') as fp:
    for pid, obj in Policy.dict_id.items():
        # 选取 Policy 数据特征
        hds = ['v01', 'z00', 'v02','v03',
              'c02', 'c03', #'c04', 
               'c06', 'c07',
               'c08',
               'c09','c12','c13',
               'c14','c15','c16'
              ]
        row = "{}".format(','.join(map(str, obj.get(hds))))
        row += ',%f'%(obj.get(['v00'])[0]/(1+obj.get(['v01'])[0]))
        #row += ',%f'%(obj.get(['v03'])[0]-(obj.get(['v02'])[0]))
        features = ['P'+hd for hd in hds]
        features.append('Pv00/Pv01')
        
        # 选取 Claim 数据特征
        if pid in Claim_feature:
            row += ',' + ','.join(map(str, Claim_feature[pid]))
        else:
            row += ',0' * 5
        features.extend(['SCv00/Pv00', 'SCv01/Pv01', 'cnt-1', 'cnt0', 'cnt1'])
        
        #选取 Customer 数据特征
        #row += ",{}".format(','.join(map(str, obj.get(hds))))
        #features = ['U'+hd for hd in hds]
        
        Customer.dict_id[pid].sort(key=lambda x:x.data['c00'], reverse=True)
        hds = ['c08', 'z03']
        features.extend('U'+hd for hd in hds)
        if len(Customer.dict_id[pid]) < 2:
            #print(temp['c08'], temp['z03'], temp['c08'], temp['z03'])
            temp = ',' + ','.join(map(str, Customer.dict_id[pid][0].get(hds)))
            row += temp+temp
        else:
            row += ',' + ','.join(map(str, Customer.dict_id[pid][0].get(hds)))
            row += ',' + ','.join(map(str, Customer.dict_id[pid][1].get(hds)))
        
            
        user = obj.series['c02']
        if 0:
            policy_vs = [0, 0, 0, 0]
            for user_pid in Customer.dict_user[user]:
                for i in range(4):
                    policy_vs[i] += float(Policy.dict_id[user_pid].get(['v0%d'%i])[0])
            policy_vs.append(len(Customer.dict_user[user])) 
            row += ',' + ','.join(map(str, policy_vs))
            features.extend(['SPv0', 'SPv1','SPv2','SPv3'])
        row += ',{}'.format(len(Customer.dict_user[user]))
        
        if 0:
            claim_vs = [0,0,0]
            cnt = 0
            for user_pid in Customer.dict_user[user]:
                if user_pid not in Claim.dict_id: continue
                for obj in Claim.dict_id[user_pid]:
                    cnt += 1
                    for i in range(3):
                        claim_vs[i] += float(obj.get(['v0%d'%i])[0])
            claim_vs.append(cnt)
            row += ',' + ','.join(map(str, claim_vs))
            
        if pid in Renewal: 
            row += ','+ str(Renewal[pid])
            row += '\n'
            fp.write(row)
            count[Renewal[pid]] += 1
            
            if pid in Claim.dict_id:
                if Renewal[pid] == 0 and random.random()<0.0:
                    fp.write(row)
                    count_add[Renewal[pid]] += 1
                elif Renewal[pid] == 1 and random.random()<0.0:
                    fp.write(row)
                    count_add[Renewal[pid]] += 1
            train_data[pid] = row
            
        else:
            row = row + '\n'
            test_data[pid] = row
        #print(row)
    def dist(train_row, test_row):
        a = np.array(list(map(float, train_row.strip().split(',')[:-1])))
        b = np.array(list(map(float, test_row.strip().split(','))))
        return np.linalg.norm(b-a)
    
    k = 0
    cnt_more = 0
    for _, test_row in test_data.items():
        break
        N = 19260
        k += 1
        if k%200 == 0:
            print(k)
        temp_row, temp_min = '', np.inf
        for _ in range(800):
            pid = random.randint(0,N-1)
            if pid in train_data:
                d = dist(train_data[pid], test_row)
                if d < temp_min:
                    temp_min, temp_row = d, train_data[pid]
        if temp_row != '':
            fp.write(temp_row)
            cnt_more += 1
    print('cnt', cnt_more)

with open(features_path, 'w') as fp:
    fp.write(','.join(features))
    
if 1:
    with open(test_file_path, 'w') as fp:
        with open('data/result.csv') as reader:
            lines = (reader.readlines())
            pids = list(map(lambda x:int(x.strip()), lines[1:]))
        for pid in pids:
            fp.write(test_data[pid])

print(count, count_add, np.array(count)+count_add)
cmd = 'cp "{}" ftp/data.csv'.format(train_file_path)
os.system(cmd)
cmd = 'cp "{}" ftp/test.csv'.format(test_file_path)
os.system(cmd)