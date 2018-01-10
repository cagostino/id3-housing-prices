import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
#import time
alldata = np.loadtxt('train.csv',dtype='<U20',delimiter=',')
testdata = np.loadtxt('test.csv',dtype='<U20',delimiter=',')
def hist(vals,n_block,attr_name,bins=None):
    if not bins:
        n,bins,patches=plt.hist(vals, bins=n_block,alpha =0.5 )
    else:
        n,bins,patches =plt.hist(vals, bins=bins,alpha=0.5)
    plt.title('Histogram of ' + attr_name)
    plt.xlabel(attr_name)
    plt.ylabel('Counts')
    plt.savefig('hists/'+attr_name+'_hist.png')
    plt.close()
    return n,bins,patches
test_pre_attr_names = testdata[0]
test_attr_names = test_pre_attr_names[1:]
pre_attr_names = alldata[0]
pre_data= np.transpose(np.copy(alldata[1:]))
test_data = np.transpose(np.copy(testdata[1:]))
test_data = np.copy(test_data)
test_attr_names =np.copy(test_pre_attr_names)
prices = np.log10(np.float64(pre_data[-1]))
pre_data= np.copy(pre_data[1:-1])
pre_attr_names = np.copy(pre_attr_names[1:-1])
neigh_ind = np.where(pre_attr_names=='Neighborhood')[0][0]
neighborhoods = np.unique(pre_data[neigh_ind])
bad_attrs= np.array(['3SsnPorch','Alley','MoSold','PoolQC',
                     'Utilities','YrSold','LandSlope',
                     'LotFrontage','MasVnrType','MasVnrArea','BsmtQual',
                     'BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
                     'Electrical','FireplaceQu','GarageType','GarageYrBlt',
                     'GarageFinish','GarageQual','GarageCond','Fence','MiscFeature',
                     'PoolArea','Street',])
                     #'1stFlrSF','2ndFlrSF','BsmtFinSF1','BsmtFinSF2','Neighborhood'])
#last row of things is getting rid of redundancies
good_attrs = np.array([[pre_attr_names[i],i]  for i in range(len(pre_attr_names)) if (pre_attr_names[i] not in bad_attrs)])
attr_names = good_attrs[:,0]
data = pre_data[np.int64(good_attrs[:,1])]
data_by_neighborhood ={} 
price_by_neighborhood = {}
for neighborhood in neighborhoods:
    neighbors_inds = np.where(data[neigh_ind,:] ==neighborhood)[0]
    neighbors_data = data[:,neighbors_inds]
    neighbors_prices = prices[neighbors_inds]
    data_by_neighborhood[neighborhood] =[neighbors_data,neighbors_data,neighbors_prices]
def fixna(attr):
    return
def plotcor(vals,prices,name):
    try:
        uvals = np.copy(np.float64(vals))
        plt.plot(uvals,prices,'o')
        plt.xlabel(name)
        plt.ylabel('log(Price)')
        plt.savefig('cor/'+name+'_cor.png') 
        plt.close()
    except:
        uniq_vals,map_ints,val_counts,map_attr = name_to_number(vals)
        plt.plot(map_attr, prices,'o')
        print(name)
        plt.xlabel(name)
        plt.ylabel('log(Price)')
        plt.savefig('cor/'+name+'_cor.png') 
        plt.close()
def loopcor():
    for i in range(len(attr_names)):
        plotcor(data[i],prices,attr_names[i])
def loophists():
    for i  in range(len(attr_names)):#len(attr_names)):
        try:
            dat= np.copy(np.float64(data[i]))
            hist(dat, 20,attr_names[i])
        except:
            dat = np.copy(data[i])
            uniq_vals,map_ints, val_counts=name_to_number(dat)
            hist(val_counts, 0, attr_names[i], bins=map_ints)
def name_to_number(attribute):
    uniq_vals, val_counts = np.unique(attribute,return_counts=True)
    map_ints = range(len(uniq_vals))
    map_attr = np.copy(attribute)
    for i in range(len(uniq_vals)):
        all_val = np.where(attribute == uniq_vals[i])[0]
        map_attr[all_val] = map_ints[i]
        intvals = map_ints[i]
        allints = np.ones_like(all_val)*intvals
    return [uniq_vals, map_ints, val_counts,map_attr]
def counts_arr(attribute):
    ''' 
    Return the unique values and their probabilities
    '''
    uniq_vals, val_counts=np.unique(attribute,return_counts=True)
    val_counts = np.float64(val_counts)
    val_probs = val_counts/np.float64(attribute.size)
    return uniq_vals, val_probs
def get_max_prob(probs):
    mxprob= np.where(probs==np.max(probs))[0][0]
    return probs[mxprob]
def entropy(attribute):
    uniq_vals , val_probs =counts_arr(attribute)
    return np.sum(val_probs *np.log2(1./val_probs))
def info_gain(attribute,labels):
    cts_arr =counts_arr(attribute)
    label_entropy = entropy(labels)
    label_given_attr = []
    if type(cts_arr[1]) != np.ndarray:
        return label_entropy
    cts_dict = dict(zip(cts_arr[0],cts_arr[1]))
    for uniq_attr_val,uniq_val_prob in cts_dict.items():
        label_given_attr.append( uniq_val_prob *entropy(labels[attribute==uniq_attr_val]))
    return label_entropy- np.sum(label_given_attr)
def best_attr(data, attribute_names,labels):
    gains=np.array([])
    for i in data:
        gains=np.append( gains, info_gain(i, labels))
    maxind = np.where(gains == np.max(gains))[0][0]
    maxgn = gains[maxind]
    att_name= attribute_names[maxind]
    print('Max Gain', maxgn)
    return att_name,maxind
class Tree:
    def __init__(self, attr_split=None,parent=None):
        self.parent=parent
        self.child=[]
        self.attr_split= attr_split
    def add_child(self,child):
        self.child.append(child)
        child.parent= self
def finite_combo(arr1,arr2):
    finds = np.where( (~np.isnan(arr1)) &(~np.isnan(arr2)))[0]
    fin_arr1 = arr1[finds]
    fin_arr2 = arr2[finds]
    return fin_arr1,fin_arr2,finds
def splitdata(attr,labels,attr_name,showplot=False,depth=0):
    ''' 
    Trying to find best spot to split at
    '''
    try:
        attr_dat =np.float64(attr)
        uniq_vals,probs = counts_arr(attr)
        typ='num'    
    except:
        uniq_vals,map_ints, val_counts, map_attr = name_to_number(attr)
        attr_dat = np.copy(np.float64(map_attr))
        typ='nom'
        print('Map Attr',map_attr)
    if typ =='nom' or len(uniq_vals) <=10:
        n_splits = len(uniq_vals)
        splits =[]
        split_ints = []
        for i in range(n_splits):
            splits.append([attr==uniq_vals[i] ])
            split_ints.append(uniq_vals[i])
        threshold_val = np.array([-99])
        threshold_type ='NA'
    elif len(uniq_vals) >2:
        try:
            labels = np.float64(labels)
            sort_by_attr = np.argsort(attr_dat)
            attr_sort = np.copy(attr_dat[sort_by_attr])
            label_sort =np.copy(labels[sort_by_attr])
            linregslopes_left = []
            linregslopes_right = []
            for i in range(len(attr_sort)//10,len(attr_sort) -len(attr_sort)//10  ):
                attr_l = attr_sort[0:i]
                attr_r = attr_sort[i:]
                lab_l = label_sort[0:i]
                lab_r = label_sort[i:]
                m_l,b_l = np.polyfit(attr_l,lab_l,1)
                m_r,b_r = np.polyfit(attr_r,lab_r,1)
                linregslopes_left.append(abs(m_l))
                #ind_left.append(i)
                linregslopes_right.append(abs(m_r))
                #ind_right.append(i)
            linregslopes_left =np.array(linregslopes_left)
            linregslopes_right =np.array(linregslopes_right)
            finite_left,finite_right,finite_inds = finite_combo(linregslopes_left,linregslopes_right)
            inv_left = 1./finite_left
            inv_right = 1./finite_right
            #lin left = where left side is linear and right is not
            linleft=finite_left*inv_right
            #linright = where right side is linear and left is not
            linright = inv_left*finite_right
            finmxleft = np.where(linleft == np.max(linleft))[0][0]
            mxleft = np.where(linregslopes_left == finite_left[finmxleft])[0]
            #transforming back to the array with infinite (if necessary)
            finmxright = np.where(linright == np.max(linright))[0][0]
            mxright = np.where(linregslopes_right == finite_right[finmxright])[0]
            #if mxleft >mxright:
            threshold_val = attr_sort[mxleft+len(attr_sort)//10]
            #else:
            #    threshold_val = attr_sort[mxright+len(attr_sort)//10]        
            if typ=='nom':
                split_l = [attr_dat <threshold_val]
                split_r = [attr_dat >=threshold_val]
                splits=[split_l,split_r]
            else:
                split_l = [attr_dat<=threshold_val]
                split_r= [attr_dat >threshold_val]
                splits=[split_l,split_r]
            threshold_type='linsplit'
            split_ints =[]
        except:
            threshold_val = np.array([np.mean(attr_dat)])
            split_l =[attr_dat < threshold_val]
            split_r = [attr_dat >= threshold_val]
            splits = [split_l,split_r]
            threshold_type = 'linsplit'
            split_ints = []
    if showplot: 
        plt.plot(attr_dat, labels,'o')
        plt.axvline(x=threshold_val,label='Threshold = '+str(threshold_val))
        plt.ylabel('log(Price))')
        plt.xlabel(attr_name)
        plt.legend()
        plt.savefig('thresholds/'+attr_name +'_threshold_depth'+str(depth)+'png')
        plt.close()
    return splits,threshold_val,threshold_type,split_ints,typ #threshold_val, mxleft, mxright
def pure(vals):
    return len(np.unique(vals))==1
def decisionTree(dat,attribute_names,labels,root=None,depth=0):
    '''
    get bestattr, threshold for split, and then values
    which align along the splits and repeat
    '''
    if dat[0] ==[]:
        root.avg_label = root.parent.avg_label
        return #root#root.parent.avg_label
    if len(dat[0]) <5:
        root.avg_label= np.mean(labels)#root#return root.parent.avg_label
        return
    bestattr,maxind= best_attr(dat,attribute_names,labels)
    if not root:
        root = Tree()
    root.split_attr =bestattr
    root.split_attr_ind=maxind
    attr_data = np.copy(dat[maxind])
    if pure(attr_data):
        root.avg_label= np.mean(labels)
        return 
    splits,threshold_val, threshold_type,split_ints,typ=splitdata(attr_data,labels,bestattr,showplot=False,depth=depth)
    trees = []
    for i in range(len(splits)):
        trees.append(Tree(parent=root))
        root.add_child(trees[i])
        root.split_ints = split_ints
        trees[i].inds_filt = splits[i][0]
        
    root.threshold_val =threshold_val
    root.threshold_type = threshold_type
    root.typ = typ
    root.avg_label= np.mean(labels)
    depth+=1
    red= [attribute_names!=bestattr]

    red_attr_names = attribute_names[red]
    red_data = dat[red]
    for tree in trees:
        decisionTree(red_data[:,tree.inds_filt],red_attr_names,labels[tree.inds_filt],root=tree,depth=depth)
    return root 
price_by_id = {}
dT = decisionTree(data,attr_names, prices)
def predictlabels(dat,attr_names,dTree,predTree = None,labels=[]):
    if len(dat[0]) == 0:
        return
    if len(labels) == 0:
        labels = np.ones_like(dat[0])
    if len(dTree.child) ==0:
        for tid in dat[0]:
            price_by_id[tid] = dTree.avg_label 
        return
    if not predTree:
        predTree = Tree()
    attr_split = dTree.split_attr 
    attr_split_ind =np.where(attr_names == attr_split)[0]
    attr_split_ints = dTree.split_ints
    attr_dat = dat[attr_split_ind][0] #these are still stored as strings
    thresh_type = dTree.threshold_type
    thresh_val = np.float64(dTree.threshold_val[0])
    try:
        attr_data =np.float64(attr_dat)
        uniq_vals,probs = counts_arr(attr_dat)
    except:
        uniq_vals,map_ints, val_counts, map_attr = name_to_number(attr_dat)
        attr_data = np.copy(np.float64(map_attr))
    typ = dTree.typ
    predTree.attr_split = attr_split
    if thresh_type=='0':
        split_l = np.where(attr_data == 0)[0]
        split_r = np.where(attr_data > 0)[0]
        splits = [split_l,split_r]
    elif thresh_type=='linsplit':
        if typ=='nom':
            split_l = np.where(attr_data <thresh_val)[0]
            split_r = np.where(attr_data >=thresh_val)[0]
            splits=[split_l,split_r]
        else:
            split_l = np.where(attr_data<=thresh_val)[0]
            split_r= np.where(attr_data >thresh_val)[0]       
            splits=[split_l,split_r]   
    elif thresh_type == 'NA':
        splits =[]
        for val in attr_split_ints:
            splits.append(np.where(attr_data==val)[0])    
    trees = []
    for i in range(len(splits)):
        trees.append(Tree(parent=predTree))
        predTree.add_child(trees[i])
        trees[i].label_inds = splits[i]
        trees[i].dtree = dTree.child[i]
        trees[i].avg_label =dTree.avg_label
        predictlabels(dat[:,trees[i].label_inds],attr_names,trees[i].dtree, predTree= trees[i],labels=labels)
    tids = dat[0]
    labs =[]
    for tid in tids:
        try:
            labs.append(price_by_id[tid])
        except:
            labs.append(np.median(list(price_by_id.values() )))
    return labs
p = predictlabels(test_data, test_attr_names,dT)
def histprices():
    plt.hist(prices, bins=10,alpha=0.75,color='g',label='Sale Prices, Train Data')
    plt.hist(p, bins=10, alpha = 0.5,color='r',label='Predicted Prices, Test Data')
    plt.legend()
    plt.xlabel('log(Price)')
    plt.ylabel('Counts')
    plt.savefig('id3_prices_hist.png')
histprices()
def writepricebyid():
    f = open('submission.csv','w+')
    f.write('Id,SalePrice\n')
    for i in range(len(test_data[0])):
        f.write(str(test_data[0][i])+','+str(10**p[i]) +'\n')
    f.close()
writepricebyid()
def tree_info(tree):
    print(tree.split_attr)
    print(tree.threshold_val)
    print(tree.avg_label)
