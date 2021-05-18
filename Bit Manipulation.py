#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 2**x = 1<<x
def Highest_Power_2_Less_than_N(n):
    x=0
    while (1<<x)<=n:
        x+=1
    return x-1


# In[27]:


n=10
Highest_Power_2_Less_than_N(n)


# In[ ]:





# In[ ]:





# In[4]:


def Right_Most_Set_Bit_Mask(n):
    rsbm=(n)&(-n)
    return "{0:b}".format(rsbm)


# In[8]:


n=10
Right_Most_Set_Bit_Mask(n)


# In[ ]:





# In[ ]:





# In[11]:


#Counting number of 1's.
#Smart Algorithm to count number of set bits.
def Kernighan(n):
    count=0
    while n!=0:
        rsbm=n&(-n)
        n-=rsbm
        count+=1
    return count


# In[3]:


#Counting number of 1's.
def CountSetBits(n):
    count=0
    while n>0:
        n=n&(n-1)
        count+=1
    return count


# In[12]:


n=51
Kernighan(51)


# In[ ]:





# In[ ]:





# In[22]:


def All_Repeating_Except_Two(arr):
    x_xor_y=0
    for i in arr:
        x_xor_y=x_xor_y^i
    x=0
    y=0
    rsbm=(x_xor_y)&(-x_xor_y)
    for i in arr:
        if rsbm&i==0:
            x=x^i
        else:
            y=y^i
            
    return x,y


# In[23]:


arr=[36,50,24,56,36,24,42,50]
All_Repeating_Except_Two(arr)


# In[ ]:





# In[ ]:





# In[24]:


def setbits(n):
    count=0
    while n!=0:
        rsbm=(n)&(-n)
        n-=rsbm
        count+=1
    return count


# In[25]:


n=6
setbits(n)


# In[ ]:





# In[ ]:





# In[34]:


def isPowerofTwo(n):
    x=0
    while (1<<x)<=n:
        if (1<<x)==n:
            return True
        x+=1
    return False


# In[33]:


def PowerOfTwo(n):
    if n==0:
        return False
    val=n&(n-1)
    if val==0:
        return True
    else:
        return False


# In[32]:


n=98
isPowerofTwo(n)


# In[ ]:





# In[ ]:





# In[41]:


def Calculate_7n_by_8(n):
    #Bring the value in terms of power of 2.
    return ((n<<3)-n)>>3


# In[43]:


n=5
Calculate_7n_by_8(n)


# In[ ]:





# In[ ]:





# In[44]:


def CountSetBits(n):
    count=0
    while n>0:
        n=n&(n-1)
        count+=1
    return count


# In[46]:


CountSetBits(7)


# In[ ]:





# In[ ]:





# In[ ]:


def Solve(n, arr):
    dp=[0 for i in range(n)]
    dp[0]=arr[0]
    dp[1]=arr[1]
    
    for i in range(2, n):
        include=arr[i]+dp[i-2]
        exclude=dp[i-1]
        dp[i]=max(include, exclude)
    return dp[=1]

