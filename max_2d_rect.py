#!/bin/python
def max2D(matrix):
    maxVal = 0
    row_accum = matrix[0]
    for index in range(len(matrix)):
        if index>=1:
            for digit in range(len(matrix[index-1])):
                if matrix[index-1][digit]==0:
                    row_accum[digit] = 0
                else:
                    row_accum[digit] += matrix[index][digit] 
        print (row_accum)
        maxVal = max(maxVal,maxHist(row_accum))
    return maxVal
        
def maxHist(lst):
    s = []
    i = 0
    max_val = 0
    while(i<len(lst)):
        if len(s)==0 or lst[i]>=lst[s[-1]]:
            s.append(i)
            i += 1
        else:
            top = s.pop()
            val = lst[top] * (i if len(s)<=0 else i-s[-1]-1)
            max_val = max(max_val,val)
    
    while(len(s)>0):
        top = s.pop()
        val = lst[top] * (i if len(s)<=0 else i-s[-1]-1)
        max_val = max(max_val,val)
    return max_val
