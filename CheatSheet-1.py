#!/usr/bin/env python
# coding: utf-8

# In[8]:


def RabinKarb(text, pattern):
    t=len(text)
    p=len(pattern)
    i=0
    j=0
    k=0
    l=[]
    while j<t:
        if k==(p-1):
            l.append(i)
            k=0
            i=j
        elif text[j]==pattern[k]:
            j+=1
            k+=1
        else:
            j+=1
            i=j
            k=0
    return l   


# In[6]:


text='AABAACAADAABAABA'
pattern='AABA'
RabinKarb(text, pattern)


# In[7]:


t='THIS IS A TEST TEXT'
p='TEST'
RabinKarb(t,p)


# In[ ]:





# In[ ]:





# In[24]:


import sys
def minSwap (arr, n, k):
    cnt=0
    for i in arr:
        if i<=k:
            cnt+=1
            
    bad=0
    for i in range(cnt):
        if arr[i]>k:
            bad+=1
            
    ans=bad   
    i=0
    j=cnt
    
    while j<n:
        if arr[j]>k:
            bad+=1
        if arr[i]>k:
            bad-=1
        ans=min(ans, bad)  
        i+=1
        j+=1
        
    return ans


# In[25]:


arr= [20,12,17]
K = 6 
minSwap(arr, len(arr), K)


# In[ ]:





# In[24]:


import sys
def sb(arr, n, x):
    # Your code goes here 
    csum=0
    cnt=0
    ans=sys.maxsize
    i=0
    j=0
    
    while j<n:
        csum+=arr[j]
        cnt+=1
        if csum>x:
            ans=min(ans, cnt)
            while csum>x:
                csum-=arr[i]
                i+=1
                cnt-=1
                if csum>x:
                    ans=min(ans, cnt)
        j+=1
        
    return ans


# In[25]:


n=6
A=[1, 4, 45, 6, 0, 19]
x = 51
sb(A, n, x)


# In[26]:


n1=8
A1 = [6, 3, 4, 5, 4, 3, 7, 9]
x1 = 16
sb(A1, n1, x1)


# In[ ]:





# In[2]:


import sys
def Distribution(n, arr, m):
    arr.sort()
    ans=sys.maxsize
    i=0
    j=m-1
    while j<n:
        value=arr[j]-arr[i]
        ans=min(ans, value)
        j+=1
        i+=1
    return ans
        


# In[4]:


n=8
a=[3, 4, 1, 9, 56, 7, 9, 12]
m=5
n1=7
a1=[7, 3, 2, 4, 9, 12, 56]
m1=3
Distribution(n1, a1, m1)


# In[ ]:





# In[ ]:





# In[28]:


def kTranscationsallowed(n, arr, k):
    dp=[[0 for j in range(n)]for i in range(k+1)]
    
    for i in range(1, k+1):
        for j in range(1, n):
            m=dp[i][j-1]
            for p in range(0, j):
                if dp[i-1][p]+(arr[j]-arr[p])>m:
                    m=dp[i-1][p]+(arr[j]-arr[p])
            dp[i][j]=m
                
    return dp[k][n-1]
        


# In[29]:


n=7
k=2
price = [2, 30, 15, 10, 8, 25, 80]
kTranscationsallowed(n, price, k)


# In[ ]:





# In[ ]:





# In[10]:


import sys
def TwoTransactionsallowed(n, arr):
    #Moving from left to right!
    max_profit_if_sold_today=0
    least_so_far=arr[0]
    dp_left=[0 for i in range(n)]
    
    for i in range(1, n):
        if arr[i]<least_so_far:
            least_so_far=arr[i]
            
        max_profit=arr[i]-least_so_far
        if max_profit>dp_left[i-1]:
            dp_left[i]=max_profit
        else:
            dp_left[i]=dp_left[i-1]
            
    #Moving from right to left!
    max_profit_if_bought_today=0
    max_so_far=arr[n-1]
    dp_right=[0 for i in range(n)]
    
    for i in range(n-2, -1, -1):
        if arr[i]>max_so_far:
            max_so_far=arr[i]
            
        max_profit=max_so_far-arr[i]
        if max_profit>dp_right[i+1]:
            dp_right[i]=max_profit
        else:
            dp_right[i]=dp_right[i+1]
            
    overall=-sys.maxsize
    for i in range(n):
        if dp_left[i]+dp_right[i]>overall:
            overall=dp_left[i]+dp_right[i]
            
    return overall


# In[12]:


n=7
price = [2, 30, 15, 10, 8, 25, 80]
TwoTransactionsallowed(n, price)


# In[ ]:





# In[ ]:





# In[3]:


def NbyKtimes(n, arr, k):
    d={}
    for i in arr:
        d[i]=d.get(i, 0)+1
    l=[]
    for i in d:
        if d[i]>(n//k):
            l.append(i)
    return l


# In[4]:


arr=[3, 1, 2, 2, 1, 2, 3, 3]
n=8
k=4
NbyKtimes(n, arr, k)


# In[ ]:





# In[ ]:





# In[50]:


def Median(n, m, l1, l2):
    i=0
    j=0
    l=[]
    while i<n and j<m:
        if l1[i]<=l2[j]:
            l.append(l1[i])
            i+=1
        elif l1[i]>l2[j]:
            l.append(l2[j])
            j+=1
    while i<n:
        l.append(l1[i])
        i+=1
    while j<m:
        l.append(l2[j])
        j+=1
        
    if (n+m)%2==0:
        return (l[(n+m)//2]+l[(n+m)//2-1])//2
    else:
        return l[(n+m)//2]


# In[53]:


ar1 = [-5, 3, 6, 12, 15]
ar2 = [-12, -10, -6, -3, 4, 10]
Median(len(ar1), len(ar2), ar1, ar2)


# In[ ]:





# In[ ]:





# In[46]:


def CommonElements(m, n, arr):
    d={}
    for i in (arr[0]):
        d[i]=1
        
    for i in range(1, m):
        for j in range(n):
            if (arr[i][j] in d) and (d[arr[i][j]]==i):
                d[arr[i][j]]+=1
                if i==m-1:
                    print(arr[i][j])
                
        


# In[47]:


m=4
n=5
mat = [[1, 2, 1, 4, 8], 
       [3, 7, 8, 1, 1], 
       [8, 7, 7, 1, 1], 
       [8, 1, 2, 1, 9]]
CommonElements(m,n,mat)


# In[ ]:





# In[ ]:





# In[33]:


import heapq
def KsmallestElement(arr, n, k):
    l=[]
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            l.append(arr[i][j])
    
    l1=l[0:k]
    l2=l[k:]
    heapq._heapify_max(l1)
    
    for i in l2:
        if i<l1[0]:
            heapq._heapreplace_max(l1, i)
            
    return l1[0]
    


# In[36]:


N = 4
mat = [[16, 28, 60, 64],
                   [22, 41, 63, 91],
                   [27, 50, 87, 93],
                   [36, 78, 87, 94]]
K = 3
KsmallestElement(mat, N, K)


# In[ ]:





# In[ ]:





# In[30]:


def Rev(l):
    i=0
    j=len(l)-1
    while i<=j:
        l[i], l[j] = l[j], l[i]
        i+=1
        j-=1
    return l

def RotateMatrix(arr):
    arr[:]=[[arr[j][i] for j in range(len(arr))]for i in range(len(arr[0]))]
    for i in range(len(arr)):
        Rev(arr[i])
    return arr
    


# In[31]:


arr=[[1, 2, 3], 
[4, 5, 6],
[7, 8, 9]]
l=[[1,2],
    [3,4]]
RotateMatrix(l)


# In[ ]:





# In[ ]:





# In[17]:


import sys
def Specific_Pair(n, arr):
    max_value=-sys.maxsize
    dp=[[0 for j in range(n)]for i in range(n)]
    
    #Initializing the max_v!
    max_v=arr[n-1][n-1]
    for j in range(n-2, -1, -1):
        if arr[n-1][j]>max_v:
            max_v=arr[n-1][j]
        dp[n-1][j]=max_v
    
    #Initializing the max_v!
    max_v=arr[n-1][n-1]
    for i in range(n-2, -1, -1):
        if arr[i][n-1]>max_v:
            max_v=arr[i][n-1]
        dp[i][n-1]=max_v
        

    for i in range(n-2, -1, -1):
        for j in range(n-2, -1, -1):
            if (dp[i+1][j+1]-arr[i][j])>max_value:
                max_value=dp[i+1][j+1]-arr[i][j]
                
            dp[i][j]=max(arr[i][j], dp[i+1][j], dp[i][j+1])
                
    return max_value
                


# In[18]:


n=5
mat = [[ 1, 2, -1, -4, -20 ], 
       [-8, -3, 4, 2, 1 ], 
       [ 3, 8, 6, 1, 3 ], 
       [ -4, -1, 1, 7, -6] , 
       [0, -4, 10, -5, 1 ]]
Specific_Pair(n, mat)


# In[ ]:





# In[33]:


import sys
def findLongestConseqSubseq(arr):
    N=len(arr)
    if N==1:
        return 1
        
    ans=-sys.maxsize
    l=sorted(list(set(arr)))
    lcs=1
    
    for i in range(0, len(l)-1):
        if l[i]+1==l[i+1]:
            lcs+=1
        else:
            lcs=1
            
        ans=max(ans, lcs)
        
    return ans


# In[34]:


a=[6,6,2,3,1,4,1,5,6,2,8,7,4,2,1,3,4,5,9,6]
findLongestConseqSubseq(a)


# In[ ]:





# In[18]:


def Rearrange(n, arr):
    p=0
    i=0
    #Partition the array around the pivot element!
    while i<n:
        if arr[i]<0:
            arr[p], arr[i] = arr[i], arr[p]
            p+=1
        i+=1
        
    neg, pos = 0, p
    
    while pos<n and neg<=pos and arr[neg]<0:
        arr[pos], arr[neg] = arr[neg], arr[pos]
        neg+=2
        pos+=1
    
    return arr   


# In[19]:


n=9
arr= [-1, 2, -3, 4, 5, 6, -7, 8, 9]
Rearrange(n, arr)


# In[ ]:





# In[ ]:





# In[ ]:





# In[49]:


import sys
def Jump(arr):
    n=len(arr)
    dp=[0 for i in range(n)]
    for i in range(n-2, -1, -1):
        steps=arr[i]
        ans=sys.maxsize
        for j in range(1, steps+1):
            if i+j<=n-1:
                ans=min(dp[i+j], ans)
        dp[i]=1+ans
        
    if dp[0]>50000:
        return -1
    else:
        return dp[0]
             


# In[50]:


A = [0, 46, 46, 0, 2, 47, 1, 24, 45, 0, 0, 24, 18, 29, 27, 11, 0, 0, 40, 12, 4, 0, 0, 0, 49, 42, 13, 5, 12, 45, 10, 0, 29, 11, 22, 15, 17, 41, 34, 23, 11, 35, 0, 18, 47, 0, 38, 37, 3, 37, 0, 43, 50, 0, 25, 12, 0, 38, 28, 37, 5, 4, 12, 23, 31, 9, 26, 19, 11, 21, 0, 0, 40, 18, 44, 0, 0, 0, 0, 30, 26, 37, 0, 26, 39, 10, 1, 0, 0, 3, 50, 46, 1, 38, 38, 7, 6, 38, 27, 7, 25, 30, 0, 0, 36, 37, 6, 39, 40, 32, 41, 12, 3, 44, 44, 39, 2, 26, 40, 36, 35, 21, 14, 41, 48, 50, 21, 0, 0, 23, 49, 21, 11, 27, 40, 47, 49 ]
Jump(A)


# In[ ]:





# In[26]:


def MinDifference(n, k, arr):
    arr=sorted(arr)
    i=0
    j=n-1
    while i<=j:
        arr[i]+=k
        arr[j]-=k
        i+=1
        j-=1
        
    return max(arr)-min(arr)
        


# In[27]:


k=5
n=10
arr=[2, 6, 3, 4, 7, 2, 10, 3, 2, 1]
MinDifference(n, k, arr)


# In[7]:


sorted(arr)


# In[ ]:





# In[59]:


def Indexes(n, arr, si, ei, x):
    mini=arr[si]
    maxi=arr[ei]
    
    while si<=ei:
        mid=si+(ei-si)//2
        if x<arr[mid]:
            ei=mid-1
        elif x>arr[mid]:
            si=mid+1
        else:
            left=mid
            right=mid
            while left>0 and arr[left]==arr[left-1]:
                left-=1
            while right<n-1 and arr[right]==arr[right+1]:
                right+=1
            return left, right
        
    return -1,-1


# In[60]:


n=9
x=125
arr=[1, 3, 5, 5, 5, 67, 123, 123, 125]
Indexes(n, arr, 0, n-1, x)


# In[ ]:





# In[ ]:





# In[5]:


def RemoveConsecutive(s):
    if len(s)==0 or len(s)==1:
        return s
    smallerOutput=RemoveConsecutive(s[1:])
    if s[0]==s[1]:
        return smallerOutput
    else:
        return s[0]+smallerOutput
    


# In[6]:


s='aaabccdefffg'
RemoveConsecutive(s)


# In[ ]:





# In[ ]:





# In[1]:


def LRS(s):
    m=len(s)
    dp=[[0 for j in range(m+1)]for i in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, m+1):
            if s[i-1]==s[j-1] and i!=j:
                dp[i][j]=1+dp[i-1][j-1]
            else:
                dp[i][j]=max(dp[i-1][j], dp[i][j-1])
                
    return (dp[m][m])


# In[2]:


x='abc'
LRS(x)


# In[ ]:





# In[ ]:





# In[7]:


from bisect import bisect_right as upper_bound
arr=[1,2,3,4,5,5,7]
upper_bound(arr, 1)


# In[ ]:





# In[ ]:





# In[1]:


def ValidShuffle(first, second, result):
    if len(first)+len(second)!=len(result):
        return False
    i,j,k=0,0,0
    while k<len(result):
        if i<len(first) and result[k]==first[i]:
            i+=1
            k+=1
        elif j<len(second) and result[k]==second[j]:
            j+=1
            k+=1
        else:
            return False
        
    return True
            


# In[4]:


f='XY'
s='12'
res='XY21'
ValidShuffle(f, s, res)


# In[ ]:





# In[ ]:





# In[8]:


def CountAndSay(n):
    if n==1:
        return '1'
    s=CountAndSay(n-1)
    
    count=1
    res=""
    i=1
    while i<len(s)+1:
        if i<len(s) and s[i]==s[i-1]:
            count+=1
        else:
            res=res+str(count)+s[i-1]
            count=1
        i+=1
    return res
            


# In[10]:


CountAndSay(8)


# In[ ]:





# In[ ]:





# In[4]:


def Check(s1, s2):
    if len(s1)!=len(s2):
        return False
    temp=s1+s2
    if temp.count(s2)>0:
        return True
    else:
        return False


# In[5]:


string1 = "AACD"
string2 = "ACDA"
Check(string1, string2)


# In[ ]:





# In[ ]:





# In[49]:


import sys
class Graph:
    def __init__(self, nvertices):
        self.nvertices=nvertices
        self.adjMat=[[0 for j in range(nvertices+1)]for i in range(nvertices+1)]
    def addEdge(self, v1, v2, d):
        self.adjMat[v1][v2]=d
        self.adjMat[v2][v1]=d
    

    def DFS(self, sv, visited, l, x):
        visited[sv]=True
        l.append(sv)
        for i in range(1, self.nvertices+1):
            if self.adjMat[sv][i]>0 and visited[i]==False:
                if x>self.adjMat[sv][i]:
                    x=self.adjMat[sv][i]
                self.DFS(i, visited, l, x)
        
    def DFS_Helper(self):
        visited=[False for i in range(self.nvertices+1)]
        c=0
        ans=[]
        for j in range(1, self.nvertices+1):
            l=[]
            x=sys.maxsize
            if visited[j]==False:
                c+=1
                self.DFS(j, visited, l, x)
                ans.append((l[0], l[-1], y))
                
        print(c)  
        for i in ans:
            print(i[0], i[1], i[2])


# In[50]:


#Input
t=int(input())
for _ in range(t):
    l=[int(i) for i in input().split()]
    n,p=l[0],l[1]
    g=Graph(n)
    for edges in range(p):
        e=[int(i) for i in input().split()]
        g.addEdge(e[0], e[1], e[2])


# In[51]:


g.DFS_Helper()


# In[ ]:





# In[ ]:





# In[5]:


def WorkBreak(word, dic):
    if len(word)==0:
        return True
    
    for i in range(1, len(word)+1):
        if word[:i] in dic and WorkBreak(word[i:], dic):
            return True
        
    return False


# In[6]:


d={ 'i', 'like', 'sam', 'sung', 'samsung', 'mobile', 'ice', 
  'cream', 'icecream', 'man', 'go', 'mango'}
word='ilikesamsun'
WorkBreak(word, d)


# In[ ]:





# In[ ]:





# In[22]:


class Meet:
    def __init__(self, s, e, pos):
        self.s=s
        self.e=e
        self.pos=pos
        
def Meetings(n, start, ending):
    l=[]
    for i in range(n):
        m=Meet(start[i], ending[i], i)
        l.append(m)
        
    l.sort(key=lambda x:x.e)
    ans=[]
    ans.append(l[0].pos)
    
    ending_time=l[0].e
    for i in range(1, n):
        if l[i].s>ending_time:
            ans.append(l[i].pos)
            ending_time=l[i].e
            
    for i in ans:
        print(i+1, end=" ")


# In[23]:


N = 8
S = [75250,50074,43659,8931,11273,
27545,50879,77924]
F = [112960,114515,81825,93424,
54316,35533,73383,160252]
Meetings(N, S, F)


# In[ ]:





# In[ ]:





# In[1]:


def gcd(a,b):
    if b==0:
        return a
    return gcd(b, a%b)


# In[6]:


gcd(1, 1337)


# In[ ]:





# In[ ]:





# In[1]:


def Loot(n, i, arr):
    if i>=n:
        return 0
    a=Loot(n, i+1, arr)
    b=arr[i]+Loot(n, i+2, arr)
    return max(a,b)


# In[15]:


def Loot_DP(n, arr):
    dp=[0 for i in range(n+1)]
    dp[0]=0
    dp[1]=arr[0]
    for i in range(2, n+1):
        dp[i]=max(dp[i-1], arr[i-1]+dp[i-2])
    return dp[n]


# In[21]:


def Loot_Pointer(n, arr):
    inc=0
    exc=0
    for i in range(0, n):
        new_exc = exc if exc>inc else inc
        
        inc=exc+arr[i]
        exc=new_exc
    return (exc if exc>inc else inc)


# In[22]:


n=6
arr=[5,5,10,100,10,5]
Loot_DP(n, arr)
Loot_Pointer(n, arr)


# In[ ]:





# In[ ]:





# In[70]:


def First(arr):
    low=0
    high=len(arr)
    while low<high:
        mid=(low+high)//2
        
        if (mid==0 or arr[mid-1]==0) and arr[mid]==1:
            return mid
        elif arr[mid]==0:
            low=mid+1
        else:
            high=mid   
    return -1
    
def Count_1(arr, row, col):
    maxi=-1
    index=-1
    for i in range(0, row):
        f=First(arr[i])
        if f!=-1:
            count=col-f
            if count>maxi:
                maxi=count
                index=i
    if index!=-1:
        return index
    else:
        return -1


# In[71]:


N = 4 
M = 4
Arr= [[0, 1, 1, 1],
      [0, 0, 1, 1],
      [1, 1, 1, 1],
      [0, 0, 0, 0]]
Count_1(Arr, N, M)


# In[72]:


n=4
m=1
a=[[0], [0], [0], [0]]
Count_1(a,n,m)


# In[ ]:





# In[63]:


def spirallyTraverse(matrix, r, c): 
    T=0
    B=r-1
    L=0
    R=c-1
    d=0
    while T<=B and L<=R:
        if d==0:
            for j in range(L, R+1):
                print(matrix[T][j], end=" ")
            T+=1
            
        elif d==1:
            for i in range(T, B+1):
                print(matrix[i][R], end=" ")
            R-=1
        
        elif d==2:
            for j in range(R, L-1, -1):
                print(matrix[B][j], end=" ")
            B-=1
            
        elif d==3:
            for i in range(B, T-1, -1):
                print(matrix[i][L], end=" ")
            L+=1
        
        d=(d+1)%4
        


# In[62]:


R = 5
C = 3
matrix = [[6,6,2],
         [28,2,12],
         [26,3,28],
         [7,22,25],
         [3,4,23]]
spirallyTraverse(matrix, R, C)


# In[ ]:





# In[ ]:





# In[13]:


def Rotate(n, arr):
    x=arr[-1]
    new=[x]+arr[0:-1]
    return new


# In[14]:


n=8
arr=[9, 8, 7, 6, 4, 2, 1, 3]
Rotate(n, arr)


# In[ ]:





# In[1]:


def move(n, arr):
    i, j = 0, n-1
    while i<=j:
        if arr[i]<0:
            i+=1
        elif arr[j]>0:
            j-=1
        else:
            arr[i], arr[j] = arr[j], arr[i]
            i+=1
            j-=1
    return arr
        


# In[2]:


l=[-12, 11, -13, -5, 6, -7, 5, -3, -6]
move(len(l), l)


# In[ ]:





# In[3]:


def poa(arr):
    if len(arr)==0:
        return 1
    else:
        return arr[0]*poa(arr[1:])


# In[5]:


arr=[1,2,3,4,5]
poa(arr)


# In[ ]:





# In[2]:


import math, operator


# In[275]:


def OldMan(n, src, aux, dest, lst):
    if n==1:
        lst.append((src, dest))
        return
    
    OldMan(n-1, src, dest, aux, lst)
    lst.append((src, dest))
    OldMan(n-1, aux, src, dest, lst)
    return


# In[292]:


def OldMan_Print(n, src, aux, dest, lst):
    if n==1:
        print(src, aux, dest)
        lst.append((src, aux, dest))
        return
    
    OldMan_Print(n-1, src, dest, aux, lst)
    print(src, aux, dest)
    lst.append((src, aux, dest))
    OldMan_Print(n-1, aux, src, dest, lst)
    return


# In[294]:


r=[]
OldMan_Print(2, 1, 2,3, r)


# In[295]:


r


# In[216]:


def spaceString(s):
    #Base Case
    if len(s)==0:
        l=[]
        l.append('')
        return l
    ans=[]
    smallerOutput=spaceString(s[1:])
    
    for i in smallerOutput:
        ans.append(s[0]+i)
    for i in smallerOutput:
        ans.append(s[0]+" "+i)
        
    return ans


# In[217]:


spaceString(s)


# In[234]:


def GF(n):
    if n==1:
        a0=0
        print(a0, end=" ")
        return 
    if n==2:
        a0=0
        a1=1
        print(a0, end=" ")
        print(a1, end=" ")
        return a0, a1
    
    a0, a1 = GF(n-1)
    print((a0**2-a1), end=" ")
    return a1, a0**2-a1


# In[235]:


GF(6)


# In[158]:


def Subsets(arr):
    #Base Case
    if len(arr)==0:
        l=[]
        l.append(0)
        return l
    
    ans=[]
    smallerOutput=Subsets(arr[1:])
    for i in smallerOutput:
        ans.append(arr[0]+i)
        
    for i in smallerOutput:
        ans.append(i)
        
    return ans


# In[160]:


a=[2,4,5]
sorted(Subsets(a))


# In[156]:


def Permutations(s):
    if len(s)==0:
        l=[]
        l.append("")
        return l
    
    ans=[]
    smallerOutput=Permutations(s[1:])
    positions = len(s)
    
    for string in smallerOutput:
        for pos in range(0, positions):
            ans.append(string[:pos]+s[0]+string[pos:])
            
    return ans


# In[157]:


s='ABC'
Permutations(s)


# In[144]:


def TowerofHanoi(n, src, aux, dest):
    #Base Case
    if n==1:
        c=0
        print('move disk', n, 'from rod', src, 'to rod', dest)
        c+=1
        return c
    
    a = TowerofHanoi(n-1, src, dest, aux)
    print('move disk', n, 'from rod', src, 'to rod', dest)
    b = TowerofHanoi(n-1, aux, src, dest)
    return 1+a+b


# In[146]:


n=3
c=0
(TowerofHanoi(n, '1', '2', '3'))


# In[22]:


def Flood_Fill(arr, m, n, x, y, k, val):
    #Base Case.
    if x>=m or y>=n or x<0 or y<0:
        return
    if arr[x][y]==val:
        arr[x][y]=k
        Flood_Fill(arr, m, n, x+1, y, k, val)
        Flood_Fill(arr, m, n, x-1, y, k, val)
        Flood_Fill(arr, m, n, x, y+1, k, val)
        Flood_Fill(arr, m, n, x, y-1, k, val)
    return 


# In[23]:


arr = [[1,1,1,0],[0,0,1,1],[1,2,2,2]]
m=3
n=4
x=1
y=2
k=3
val=arr[x][y]
Flood_Fill(arr, m, n, x, y, k, val)


# In[49]:


def Printing(n):
    if n==1:
        print(n, end=" ")
        return 

    Printing(n-1)
    print(n, end=" ")
    return


# In[50]:


Printing(10)


# In[81]:


def Paths(m, n, i, j):
    #Base Case
    if i==m-1 and j==n-1:
        return 1
    
    #Edges Cases.
    if i>=m or j>=n:
        return 0
        
    smallerOutput1 = Paths(m, n, i+1, j)
    smallerOutput2 = Paths(m, n, i, j+1)
    
    return smallerOutput1+smallerOutput2
        


# In[82]:


Paths(3,3,0,0)


# In[105]:


def Print_Pattern(n):
    if n==0 or n<0:
        print(n, end=" ")
        return 
    
    print(n, end=" ")
    Print_Pattern(n-5)
    print(n, end=" ")
    


# In[106]:


Print_Pattern(10)


# In[299]:


def reverse(n):
    rev=0
    while n!=0:
        r=n%10
        rev = rev*10+r
        n=n//10
        
    return rev


# In[300]:


reverse(159)


# In[29]:


def longestWord(words):
    words = sorted(words)
    i=1
    while i<len(words):
        curr=words[i]
        t=curr.find(words[i-1])
        if t==-1:
            break
        i+=1
    if i>len(words):
        return ""
    else:
        return words[i-1]


# In[33]:


w =  []
longestWord(w)


# In[10]:


def distributeCandies(candies):
        n = len(candies)
        d={'sister':[],
          'brother':[]}
        
        for c in candies:
          if len(d['sister'])<n//2 and (c not in d['sister']):
            d['sister'].append(c)
          else:
            d['brother'].append(c)
            
        return len(d['sister'])


# In[11]:


candies = [1,1,2,2,3,3]
distributeCandies(candies)


# In[43]:


def topKFrequent(words, k):
    d={}
    for name in words:
        d[name]=d.get(name,0)+1
    l = sorted(d.items(), key=operator.itemgetter(1), reverse=True)[:k]
    return l


# In[44]:


l = ["i", "love", "leetcode", "i", "love", "coding"]
a = topKFrequent(l, 2)


# In[51]:


d={}
for i in a:
    if i[0] not in d:
        d[i[0]]=True
d


# In[9]:


def Count(n):
    count=0
    for i in range(1, n):
        if n%i==0:
            count+=1
    return count
        


# In[10]:


Count(4)


# In[41]:


import math
x = 5
no_of_bits = int(math.floor(math.log(x)//math.log(2)))+1
((1 << no_of_bits) - 1) ^ x


# In[36]:


a = ["cool","lock","cook"]
commonChars(a)


# In[2]:


l =  [3,2,3,1,2,4,5,5,6]
a = sorted(list(set(l)), reverse=True)
a


# In[27]:


def Substring(a, b):
    c=""
    i=0
    while i <= (len(b)//len(a))+1:
        if b in c:
            return i
        c+=a
        i+=1
    return -1


# In[28]:


a = "abcd"
b = "cdabcdab"
Substring(a, b)


# In[29]:


s='abcd'
for i,ch in enumerate(s):
    print(i, ch)


# In[20]:


def rob(nums):
      l = len(nums)
      dp = [0 for i in range(l+1)]
      dp[0]=0
      dp[1]=nums[0]

      for i in range(2, l+1):
        dp[i] = max(nums[i-1]+dp[i-2], dp[i-1])
        
      return dp[l]


# In[19]:


a = [1,2,3,1]
rob(a)


# In[14]:


I = ["flower","flow","flight"]
s = sorted(I, key = len)
x = s[0]
ans = 1000000
for i in range(1, len(s)):
    k = 0
    string = s[i]
    for ch in string:
        if k<len(x) and ch == x[k]:
            output = k
            k+=1
        else:
            break
    ans = min(ans, output)
x[0:ans+1]


# In[19]:


import sys
def prefix(arr):
    s = sorted(arr, key = len)
    x = s[0]
    ans = sys.maxsize
    output = sys.maxsize
    for i in range(1, len(s)):
        k = 0
        string = s[i]
        for ch in string:
            if k<len(x) and ch == x[k]:
                output = k
                k+=1
            else:
                break
        ans = min(ans, output)
        
    if ans==sys.maxsize:
        return ""
    else:
        return x[0:ans+1]
    


# In[42]:


def longestCommonPrefix(strs):
  longest = ""
  if len(strs)==0:
    return longest
  strs = sorted(strs, key=len)
  first = strs[0]
    
  for index in range(len(first)):
    count=0
    for i in range(1, len(strs)):
      if first[index]!=strs[i][index]:
        return longest
      else:
        count+=1
    if count==len(strs)-1:
      longest = longest+first[index]
      
  return longest


# In[43]:


I = ["aa", 'a']
longestCommonPrefix(I)


# In[29]:


I[0][1]


# In[45]:


s1 = 'hello'
s2 = 'ab'
s1.find(s2)


# In[60]:


def climbStairs(n, dp):
    if n==1:
        return 1
    if n==2:
        return 2
    
    if dp[n-1]==-1:
        smallerOutput1 = climbStairs(n-1, dp)
        dp[n-1]=smallerOutput1
    else:
        smallerOutput1 = dp[n-1]
        
    if dp[n-2]==-1:
        smallerOutput2 = climbStairs(n-2, dp)
        dp[n-2]=smallerOutput2
    else:
        smallerOutput2 = dp[n-2]
        
    return smallerOutput1 + smallerOutput2


# In[64]:


n = 38
dp = [-1 for i in range(n+1)]
climbStairs(n, dp)


# In[85]:


a = '1111'
b = '1111'
addBinary(a,b)


# In[ ]:





# In[90]:


def getMaximumGold(grid):
      m=len(grid)
      n=len(grid[0])
      visited=[[False for j in range(n)]for i in range(m)]
      ans=-1
      for i in range(m):
        for j in range(n):
          if grid[i][j]:
            v=getGold(grid, i, j, m, n, visited)
            ans=max(ans,v)
      return (ans)
            
            
            
def getGold(grid, i, j, m, n, visited):
  #Base Case
  if i<0 or j<0 or i>=m or j>=n or grid[i][j]==0 or visited[i][j]==True:
    return 0
  
  visited[i][j]=True
  u=getGold(grid, i+1, j, m, n, visited)
  d=getGold(grid, i-1, j, m, n, visited)
  l=getGold(grid, i, j-1, m, n, visited)
  r=getGold(grid, i, j+1, m, n, visited)
  
  visited[i][j]=False

  return max(u,d,l,r)+grid[i][j]
  


# In[91]:


g=[[1,0,7],
   [2,0,6],
   [3,4,5],
   [0,3,0],
   [9,0,20]]


# In[92]:


getMaximumGold(g)


# In[ ]:





# In[52]:


def partition(arr, si, ei):
    pivot = arr[si]
    #Counting the number of elements smaller than pivot
    count=0
    i=si+1
    while i<=ei:
        if arr[i]<pivot:
            count+=1
        i+=1
    #Swapping the pivot element to its right place.
    arr[si+count], arr[si] = arr[si], arr[si+count]
    pivot_index = si+count
    #Swapping the elements to their correct positions
    while si<ei:
        if arr[si]<pivot:
            si+=1
        elif arr[ei]>=pivot:
            ei-=1
        else:
            t = arr[si]
            arr[si]=arr[ei]
            arr[ei]=t
            si+=1
            ei-=1

    return pivot_index


# In[95]:


def Partition(arr, si, ei):
    #Pick the pivot element
    p=arr[si]
    
    #Count how mant elements are smaller than pivot element
    cnt=0
    i=si+1
    while i<=ei:
        if arr[i]<p:
            cnt+=1
        i+=1
            
    #Put the pivot at correct postion.     
    arr[si], arr[si+cnt] = arr[si+cnt], arr[si]
    
    #Now partition the array!
    while si<ei:
        if arr[si]<p:
            si+=1
        elif arr[ei]>p:
            ei-=1
        else:
            arr[si], arr[ei] = arr[ei], arr[si]
            si+=1
            ei-=1
            
    return (si+cnt)
    
def QuickSort(arr, si, ei):
    if si>ei:
        return 
    
    pivot=partition(arr, si, ei)
    
    QuickSort(arr, si, pivot-1)
    QuickSort(arr, pivot+1, ei)


# In[96]:


arr=[3,4,5,1,2,9,8,7]
QuickSort(arr, 0, len(arr)-1)


# In[97]:


arr


# In[ ]:





# In[ ]:





# In[14]:


def arrange_amplifiers(n, arr):
    one=[]
    other=[]
    for i in range(n):
        if arr[i]==1:
            one.append(arr[i])
        else:
            other.append(arr[i])
        
    if len(other)==2 and ((other[0]==2 and other[1]==3) or (other[0]==3 and other[1]==2)):
        other.sort()
        return one+other
    else:
        other.sort(reverse=True)
        return one+other


# In[16]:


n=7
arr=[1, 3, 4, 1, 1, 1, 2] 
print(*arrange_amplifiers(n, arr))


# In[ ]:





# In[ ]:





# In[32]:


def Wines(n, arr):
    i=0
    j=1
    cost=0
    
    while j<n:
        if arr[i]*arr[j]<0:
            if abs(arr[i])<abs(arr[j]):
                cost+=(j-i)*abs(arr[i])
                arr[j]=arr[i]+arr[j]
                arr[i]=0
                i=j
                j+=1
            else:
                cost+=(j-i)*abs(arr[j])
                arr[i]=arr[i]+arr[j]
                arr[j]=0
                j+=1
            
        else:
            bottles=abs(arr[i])
            arr[j]=arr[j]+arr[i]
            arr[i]=0
            cost+=(j-i)*bottles
            i=j
            j+=1
            
    return cost
            
            


# In[ ]:





# In[ ]:





# In[41]:


def DIEHARD(H, A):
    c=1
    t=0
    
    while H>0 and A>0:
        if c%2==1:
            H+=3
            A+=2
            t+=1
        else:
            if H>5 and A>10:
                H-=5
                A-=10
                t+=1
            else:
                H-=20
                A+=5
                t+=1
        c+=1
                
    return t-1


t=int(input())
for _ in range(t):
    l=[int(i) for i in input().split()]
    H,A=l[0],l[1]
    print(DIEHARD(H,A))


# In[42]:


H=2
A=10
DIEHARD(H, A)

