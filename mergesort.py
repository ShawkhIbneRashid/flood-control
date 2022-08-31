class MergeSort():

    
    
    def merge(self, list_im_arr, arr, l, m, r):
        n1 = m - l + 1
        n2 = r - m
        
        L = [0] * (n1)
        LIm = [0] * (n1)
        R = [0] * (n2)
        RIm = [0] * (n2)
        
        for i in range(0, n1):
            L[i] = arr[l + i]
            LIm[i] = list_im_arr[l + i]
        
        for j in range(0, n2):
            R[j] = arr[m + 1 + j]
            RIm[j] = list_im_arr[m + 1 + j]
        
        
        i = 0	 
        j = 0	 
        k = l	 
        
        while i < n1 and j < n2:
            if L[i] <= R[j]:
                arr[k] = L[i]
                list_im_arr[k] = LIm[i]
                i += 1
            else:
                arr[k] = R[j]
                list_im_arr[k] = RIm[j]
                j += 1
                k += 1
        
        while i < n1:
            arr[k] = L[i]
            list_im_arr[k] = LIm[i]
            i += 1
            k += 1
        
        
        while j < n2:
            arr[k] = R[j]
            list_im_arr[k] = RIm[j]
            j += 1
            k += 1
 
    
    def mergeSort(self, list_im_arr, arr, l, r):
        if l < r:
            m = l+(r-l)//2
            self.mergeSort(list_im_arr, arr, l, m)
            self.mergeSort(list_im_arr, arr, m+1, r)
            self.merge(list_im_arr, arr, l, m, r)

