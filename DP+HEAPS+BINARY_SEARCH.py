#!/usr/bin/env python
# coding: utf-8

# In[37]:


import sys
def MinCostOranges(n, cost, W):
    wt=[]
    value=[]
    for i in range(n):
        if cost[i]!=-1:
            value.append(cost[i])
            wt.append(i+1)
            
    dp=[[0 for j in range(W+1)]for i in range(n+1)]
    
    for j in range(0, W+1):
        dp[0][j]=sys.maxsize
    for i in range(1, n+1):
        dp[i][0]=0
        
    for i in range(1, n+1):
        for j in range(1, W+1):
            if wt[i-1]<=j:
                dp[i][j]=min(value[i-1]+dp[i][j-wt[i-1]] , dp[i-1][j])
            else:
                dp[i][j]=dp[i-1][j]
        
    return dp[n][W]


# In[38]:


n=5
arr = [20, 10, 4, 50, 100]
W = 5
MinCostOranges(n, arr, W)


# In[ ]:





# In[ ]:





# In[65]:


def Removals(arr, k, i, j, dp):
    #Base Case
    if i>=j:
        return 0
    if arr[j]-arr[i]<=k:
        return 0
    
    if dp[i+1][j]==-1:
        a1=Removals(arr, k, i+1, j, dp)
    else:
        a1=dp[i+1][j]
        
    if dp[i][j-1]==-1:
        a2=Removals(arr, k, i, j-1, dp)
    else:
        a2=dp[i][j-1]
        
    return 1+min(a1, a2)
    


# In[66]:


a = [1, 5, 6, 2, 8]
k = 2
a.sort()
dp=[[-1 for j in range(len(a)+1)]for i in range(len(a)+1)]
Removals(a, k, 0, len(a)-1, dp)


# In[ ]:





# In[ ]:





# In[69]:


def smallest_sum(n, arr):
    csum=arr[0]
    osum=arr[0]
    
    for i in range(1, n):
        if csum<=0:
            csum+=arr[i]
        else:
            csum=arr[i]
            
        if csum<=osum:
            osum=csum
            
    return osum


# In[72]:


n=5
a=[2, 6, 8, 1, 4]
smallest_sum(n, a)


# In[ ]:





# In[ ]:





# In[1]:


def Longest_Alternating_Sub(n, arr):
    dp=[[1 for j in range(2)]for i in range(n)]
    ans=0
    
    for i in range(1, n):
        for j in range(0, i):
            if arr[i]>arr[j] and dp[i][0]<dp[j][1]+1:
                dp[i][0]=dp[j][1]+1
            elif arr[i]<arr[j] and dp[i][1]<dp[j][0]+1:
                dp[i][1]=dp[j][0]+1
                
        ans=max(ans, dp[i][0], dp[i][1])
        
    return ans
    


# In[2]:


n=10
a=[1,17,5,10,13,15,10,5,16,8]
Longest_Alternating_Sub(n, a)


# In[ ]:





# In[ ]:





# In[35]:


import sys
def isPalindrone(s):
    i=0
    j=len(s)-1
    while i<=j:
        if s[i]!=s[j]:
            return False
        i+=1
        j-=1
    return True

def Palindronic_Partitioning(s, i, j, dp):
    #Base Case
    if i>=j:
        return 0
    if isPalindrone(s):
        return 0
    
    ans=sys.maxsize
    #Find the loop of k
    for k in range(i, j):
        if dp[i][k]==-1:
            a1=Palindronic_Partitioning(s, i, k, dp)
            dp[i][k]=a1
        else:
            a1=dp[i][k]
        
        if dp[k+1][j]==-1:
            a2=Palindronic_Partitioning(s, k+1, j, dp)
            dp[k+1][j]=a2
        else:
            a2=dp[k+1][j]
            
        total=1+a1+a2
        ans=min(ans, total)
        
    return ans
        
    


# In[36]:


s='abccbc'
dp=[[-1 for j in range(len(s)+1)]for i in range(len(s)+1)]
Palindronic_Partitioning(s, 0, len(s)-1, dp)


# In[ ]:





# In[ ]:





# In[8]:


def Optimal_Game_Strategy(n, arr):
    dp=[[0 for j in range(n)]for i in range(n)]
    for g in range(0, n):
        for i, j in zip(range(0, n) , range(g, n)):
            if g==0:
                dp[i][j]=arr[i]
            elif g==1:
                dp[i][j]=max(arr[i], arr[j])
            else:
                o1=arr[i]+min(dp[i+2][j], dp[i+1][j-1])
                o2=arr[j]+min(dp[i+1][j-1], dp[i][j-2])
                dp[i][j]=max(o1, o2)
                
    return dp[0][n-1]


# In[9]:


n=4
a=[8,15,3,7]
Optimal_Game_Strategy(n ,a)


# In[ ]:





# In[ ]:





# In[34]:


import sys
def Optimal_BST(n, arr, freq):
    dp=[[0 for j in range(n)]for i in range(n)]
    for g in range(0, n):
        for i, j in zip(range(0, n), range(g, n)):
            if g==0:
                dp[i][j]=freq[i]
            elif g==1:
                dp[i][j]=min(1*freq[i]+2*freq[j] , 2*freq[i]+1*freq[j])
            else:
                #Now Apply MCM.
                fs=0
                ans=sys.maxsize
                for x in range(i, j+1):
                    fs+=freq[x]
                for k in range(i, j):
                    left=dp[i][k-1]
                    right=dp[k+1][j]
                    total=left+right+fs
                    
                    ans=min(ans, total)
                    
                dp[i][j]=ans
                
    return dp[0][n-1]
                    


# In[35]:


n=3
keys = [10, 12, 20] 
freq = [34, 8, 50]
Optimal_BST(n, keys, freq)


# In[ ]:





# In[ ]:





# In[ ]:





# In[84]:


def Game_Strategy_Rec(n, x, y):
    if n<=0:
        return False
    if n==1:
        return True
    
    a1=Game_Strategy(n-1, x, y)
    a2=Game_Strategy(n-x, x, y)
    a3=Game_Strategy(n-y, x, y)
    
    if (a1==True) and (a2==True) and (a3==True):
        return False
    else:
        return True
    


# In[85]:


n = 4
x = 2
y = 3
Game_Strategy_Rec(n, x, y)


# In[86]:


def Game_Strategy_DP(n, x, y):
    dp=[0 for i in range(n+1)]
    dp[0]=False
    dp[1]=True
    
    for i in range(2, n+1):
        if (i-1)>=0 and dp[i-1] is False:
            dp[i]=True
        elif (i-x)>=0 and dp[i-x] is False:
            dp[i]=True
        elif (i-y)>=0 and dp[i-y] is False:
            dp[i]=True
        else:
            dp[i]=False
            
    return dp[n]


# In[ ]:





# In[ ]:





# In[89]:


def dearrangement(n):
    dp=[0 for i in range(n+1)]
    dp[0]=0
    dp[1]=0
    dp[2]=1
    
    for i in range(3, n+1):
        dp[i]=(i-1)*dp[i-1]+(i-1)*dp[i-2]
    return dp[n]


# In[90]:


dearrangement(4)


# In[ ]:





# In[ ]:





# In[122]:


def maxProfit(k, n, arr):
    dp=[[0 for j in range(n+1)]for i in range(k+1)]
    for i in range(1, k+1):
        for j in range(1, n):
            dp[i][j] = max(dp[i-1][j-1]+(arr[j]-arr[j-1]) , dp[i][j-1])
    return dp[k][n-1]
                    


# In[124]:


k = 3
price = [12, 14, 17, 10, 14, 13, 12, 15] 
maxProfit(k, len(price), price)


# In[ ]:





# In[ ]:





# In[8]:


def getCount(n):
    dp=[[0 for j in range(10)]for i in range(n+1)]
    allowed=[[] for i in range(10)]
    
    allowed[0]=[0,8]
    allowed[1]=[1,2,4]
    allowed[2]=[1,2,3,5]
    allowed[3]=[2,3,6]
    allowed[4]=[1,4,5,7]
    allowed[5]=[2,4,5,6,8]
    allowed[6]=[3,5,6,9]
    allowed[7]=[4,7,8]
    allowed[8]=[5,7,8,9,0]
    allowed[9]=[6,8,9]
    
    for i in range(1, n+1):
        for j in range(10):
            if i==1:
                dp[i][j]=1
            else:
                s=0
                key=allowed[j]
                for x in key:
                    s+=dp[i-1][x]
                dp[i][j]=s
                
    return sum(dp[n])


# In[9]:


n=2
getCount(n)


# In[ ]:





# In[ ]:





# In[19]:


import sys
def Kadanes(arr):
    n=len(arr)
    csum=arr[0]
    osum=arr[0]
    for i in range(1, n):
        if csum>0:
            csum+=arr[i]
        else:
            csum=arr[i]
            
        if csum>osum:
            osum=csum
    return osum
            
      
def Helper(n, m, mat):
    ans=-sys.maxsize
    temp=[0 for i in range(n)]
    
    for left in range(0, m):
        for i in range(n):
            temp[i]=0
            
        for right in range(left, m):
            for i in range(n):
                temp[i]+=mat[i][right]
                
            x=Kadanes(temp)
            ans=max(ans,x)
        
    return ans


# In[20]:


n=4
m=5
mat=[[1, 2, -1, -4, -20],
     [-8, -3, 4, 2, 1], 
      [3, 8, 10, 1, 3],
      [-4, -1, 1, 7, -6]]
Helper(n, m, mat)


# In[ ]:





# In[ ]:





# In[5]:


import sys
def Maximum_Subarray(arr, k):
    n=len(arr)
    i=0
    j=0
    
    maximum=-sys.maxsize
    res=[]
    
    while j<n:
        if arr[j]>maximum:
            maximum=arr[j]   
        if (j-i+1)==k:
            res.append(maximum)
            i+=1
        j+=1
        
    return res


# In[8]:


arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
K = 3
Maximum_Subarray(arr, K)


# In[ ]:





# In[ ]:





# In[31]:


import heapq
def kLargest(arr, n, k):
    li=[]
    for i in range(k):
        li.append(arr[i])   
        
    heapq._heapify_max(li)
        
    for i in range(k, n):
        top=li[0]
        if arr[i]<top:
            heapq._heapreplace_max(li, arr[i])
            
    return li[0]


# In[33]:


N = 6
K = 3
Arr = [7, 10, 4, 3, 20, 15]
kLargest(Arr, N, K)


# In[ ]:





# In[ ]:





# In[43]:


import heapq
def merge(numbers):
    k=len(numbers)
    li=[]
    res=[]
    
    for i in range(k):
        pair=(numbers[i][0], i, 0)
        li.append(pair)
        
    heapq.heapify(li)
        
    while len(li)!=0:
        #Pop the top element!
        top=heapq.heappop(li)
        
        #Store all the parameters
        value=top[0]
        list_num=top[1]
        element_num=top[2]
        
        #Store the result
        res.append(value)
        
        #Now insert the value
        if element_num<(len(numbers[list_num])-1):
            pair=(numbers[list_num][element_num+1] , list_num, element_num+1)
            li.append(pair)
            
        #Heapify the list
        heapq.heapify(li)
        
    return res
            
            


# In[44]:


k = 4
arr=[[1,2,3,4], [2,2,3,4],
     [5,5,6,6], [7,8,9,9]]
merge(arr)


# In[ ]:





# In[ ]:





# In[48]:


def Heapify_Down(n, l, i):
    parent_index=i
    
    while parent_index<n:
        left_child=(2*parent_index)+1
        right_child=(2*parent_index)+2
        
        if left_child>=n:
            break
        
        if right_child<n:
            if l[left_child]>l[right_child]:
                child_index=left_child
            else:
                child_index=right_child
        else:
            child_index=left_child
            
            
        if l[parent_index]<l[child_index]:
            l[parent_index], l[child_index] = l[child_index], l[parent_index]
            parent_index=child_index
            
            
        else:
            break
            
    return 


def Merge(n, arr):
    for i in range(n-1, -1, -1):
        Heapify_Down(n, arr, i)
    return arr


# In[49]:


n=4
m=3
a1=[10, 5, 6, 2]
a2=[12, 7, 9]
arr=a1+a2
Merge(n+m, arr)


# In[ ]:





# In[ ]:





# In[ ]:





# In[1]:


import sys
def Matrix_Zero_Sum(arr):
    n=len(arr)
    d={}
    ans=-sys.maxsize
    s=0
    #Sum zero is at -1 index!
    d[0]=-1
    
    for i in range(n):
        s+=arr[i]
        
        if s in d:
            length=i-d[s]
            ans=max(length, ans)
        else:
            d[s]=i
            
    return ans


def Helper(r, c, mat):
    ans=-sys.maxsize
    
    for row in range(0, r):
        temp=[0 for k in range(c)]
        
        for i in range(row, r):
            for j in range(0, c):
                temp[col]+=mat[i][j]
            x=Matrix_Zero_Sum(temp)
            area=(x)*(i-row+1)
            ans=max(ans, area)
            
    return ans
            


# In[ ]:





# In[ ]:





# In[10]:


import heapq
def reorganizeString(s):
    d={}
    res=""
    li=[]
        
    for ch in s:
        d[ch]=d.get(ch,0)+1
          
    for ch, freq in d.items():
        heapq.heappush(li, [-freq, ch])
          
          
    while len(li)>1:
        #Popping the topmost element
        item1=heapq.heappop(li)
        item2=heapq.heappop(li)
        v1, c1 = -item1[0], item1[1]
        v2, c2 = -item2[0], item2[1]
          
        #Adding in the result.
        res+=c1
        res+=c2
          
        if v1>1:
            heapq.heappush(li, [-(v1-1), c1])
        if v2>1:
            heapq.heappush(li, [-(v2-1), c2])
            
        
    if len(li)>0:
        item=heapq.heappop(li)
        val, char = -item[0], item[1]
        if val>1:
            return ""
        else:
            res+=char
            
    return "".join(res)


# In[12]:


s='aaabc'
reorganizeString(s)


# In[ ]:





# In[ ]:





# In[35]:


import heapq
def MinCostofRopes(n, arr):
    l=[]
    for i in arr:
        heapq.heappush(l, i)
    cost=0
    
    while len(l)>1:
        a=heapq.heappop(l)
        b=heapq.heappop(l)
        new_rope=a+b
        cost+=new_rope
        heapq.heappush(l, new_rope)
        
    return cost


# In[36]:


n=4
arr = [4, 3, 2, 6]
MinCostofRopes(n, arr)


# In[ ]:





# In[ ]:





# In[36]:


def searchInrotated(arr, l, h, key):
    if l>h:
        return -1
    
    mid=l+(h-l)//2
    if arr[mid]==key:
        return mid
    
    if arr[l]<=arr[mid]:
        if key>=arr[l] and key<=arr[mid]:
            return searchInrotated(arr, l, mid-1, key)
        else:
            return searchInrotated(arr, mid+1, h, key)
            
    else:
        if key>=arr[mid] and key<=arr[h]:
            return searchInrotated(arr, mid+1, h, key)
        else:
            return searchInrotated(arr, l, mid-1, key)


# In[38]:


arr = [5, 6, 7, 8, 9, 10, 1, 2, 3]
k = 7
searchInrotated(arr, 0, len(arr)-1, k)


# In[ ]:





# In[ ]:





# In[1]:


def findpivot(arr, l, h):
    if l>h:
        return -1
    if l==h:
        return l
    
    mid=l+(h-l)//2
    if l<mid and arr[mid]<arr[mid-1]:
        return mid-1
    if h>mid and arr[mid]>arr[mid+1]:
        return mid
    
    #If first half is not sorted, then pivot is found in first half
    if arr[l]>=arr[mid]:
        return findpivot(arr, l, mid-1)
    #Otherwise, it is found in second half!
    else:
        return findpivot(arr, mid+1, h)


# In[2]:


arr = [5, 6, 7, 8, 9, 10, 1, 2, 3]
findpivot(arr, 0, len(arr)-1)


# In[ ]:





# In[ ]:





# In[114]:


import sys
def findNum(n):
    l=0
    h=n*5
    ans=sys.maxsize
    
    while l<=h:
        mid=l+(h-l)//2
        trailing_mid=Trailingzeros(mid)
            
        if trailing_mid<n:
            l=mid+1
        else:
            ans=mid
            h=mid-1
                
    return ans

def Trailingzeros(x):
    z=0
    while x!=0:
        z+=x//5
        x=x//5
    return z


# In[115]:


findNum(1)


# In[ ]:





# In[ ]:





# In[120]:


import sys
def WoodLength(n, arr, mid, m):
    l=0
    for i in range(n):
        if arr[i]>=mid:
            l+=(arr[i]-mid)
        
    return l

def EKO(n, arr, m):
    low=0
    high=max(arr)
    ans=sys.maxsize
    
    while low<=high:
        mid=low+(high-low)//2
        
        x=WoodLength(n, arr, mid, m)
        if x>m:
            low=mid+1
        else:
            ans=mid
            high=mid-1
            
    return ans
            


# In[122]:


n=5
m=20
arr=[4, 42, 40, 26, 46]
EKO(n, arr, m)


# In[ ]:





# In[ ]:





# In[161]:


def WeightedJob(n, arr):
    l=sorted(arr, key=lambda x:x[1])
    dp=[0 for i in range(n)]
    #Base Case
    dp[0]=arr[0][2]
    
    for i in range(1, n):
        include=arr[i][2]
        last=-1
        low=0
        high=i-1
        #SEARCH WHICH JOB WE CAN PICK BEFORE IT
        while low<=high:
            mid=low+(high-low)//2
            if arr[i][0]>=arr[mid][1]:
                last=mid
                low=mid+1
            else:
                high=mid-1
                
        if last!=-1:
            include+=dp[last]
            
        exclude=dp[i-1]
        dp[i]=max(include, exclude)
        
    return dp[n-1]
                
        


# In[162]:


n=4
j = [(1, 2, 50), (3, 5, 20),  (6, 19, 100), (2, 100, 200)]
WeightedJob(n, arr)


# In[ ]:





# In[ ]:





# In[2]:


def MakePratha(n, arr, mid, P):
    paratha=0
    time=0
    
    for i in range(0, n):
        time=arr[i]
        j=2
        while time<=mid:
            paratha+=1
            time=time+(j*arr[i])
            j+=1
        if paratha>=P:
            return True
        
    return False

def RotiPratha(P, n, arr):
    low=0
    high=10**8
    
    while low<=high:
        mid=low+(high-low)//2
        if MakePratha(n, arr, mid, P):
            ans=mid
            high=mid-1
        else:
            low=mid+1
            
    return ans
        


# In[3]:


p=10
n=4
arr=[1, 2, 3, 4]
RotiPratha(p, n, arr)


# In[ ]:





# In[ ]:





# In[8]:


def Double_Helix(a1, a2):
    n=len(a1)
    m=len(a2)
    i=0
    j=0
    s1=0
    s2=0
    res=0
    while i<n and j<m:
        if a1[i]<a2[j]:
            s1+=a1[i]
            i+=1
        elif a1[i]>a2[j]:
            s2+=a2[j]
            j+=1
        else:
            res+=max(s1,s2)
            res+=a2[j]
            i+=1
            j+=1
            s1=0
            s2=0
    
    while i<n:
        s1+=a1[i]
        i+=1
    while j<m:
        s2+=a2[j]
        j+=1
    return res+max(s1, s2)
            


# In[9]:


a1= [3, 5, 7, 9, 20, 25, 30, 40, 55, 56, 57, 60, 62]
a2= [1, 4, 7, 11, 14, 25, 44, 47, 55, 57, 100]
Double_Helix(a1, a2)


# In[ ]:





# In[34]:


def Merge(arr1, arr2):
    n=len(arr1)
    m=len(arr2)
    res=[]
    i=0
    j=0
    
    while i<n and j<m:
        if arr1[i]<arr2[j]:
            res.append(arr1[i])
            i+=1
        elif arr1[i]>arr2[j]:
            res.append(arr2[j])
            j+=1
        else:
            res.append(arr1[i])
            res.append(arr2[j])
            i+=1
            j+=1
            
    while i<n:
        res.append(arr1[i])
        i+=1
    while j<m:
        res.append(arr2[j])
        j+=1
        
    return res
    
def Merge_Sort(arr):
    if len(arr)==1:
        return arr
    
    mid=len(arr)//2
    a1=arr[0:mid]
    a2=arr[mid:]
    
    l1=Merge_Sort(a1)
    l2=Merge_Sort(a2)
    
    return Merge(l1,l2)
    


# In[36]:


arr=[12, 11, 13, 5, 6, 7 ]
Inplace_Merge_Sort(len(arr), arr)

