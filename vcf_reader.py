# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 18:29:36 2022

@author: Niels

"""

#%%
import numpy as np

class vcf_reader():
    '''Convert vcf format SNP data to a format usable by neural networks.
    Assumes GT format'''
    def __init__(self, vcf_file):
        #Count the number of lines in the file
        file_len = len(open(vcf_file, 'r').readlines())
        #Get the file object
        with open(vcf_file, 'r') as vcf:
            #Read lines until the column names appear
            #and store the preceeding lines in self.header
            self.header = list()
            line = line = vcf.readline()
            line = line.strip()
            while line[:6] != '#CHROM':
                self.header.append(line)
                line = vcf.readline()
                line = line.strip()
                
            self.info_cols = line.split('\t')[:9] #Store column names
            
            self.samples = line.split('\t')[9:] #Store samples names
            
            self.n_snp = file_len-len(self.header)-1 #Number of snps in the file
            
            self.snp_info = np.zeros((self.n_snp,
                                     len(self.info_cols)),
                                     dtype='O') #An array to store the info columns
            
            self.n_samples = len(self.samples) #Number of samples in file
            
            self.data = np.zeros((self.n_samples, self.n_snp),
                                 dtype=np.int8) #Tensor for numeric genotypes
                        
            minor_allele_combs = [f'{x}'.join([str(a),str(b)]) for a in range(1,10) for b in range(1,10) for x in ['/','|']]
            
            major_minor_combs = [f'{i}{k}0' for i in range(1,10) for k in ['|','/']]
            major_minor_combs += [g[::-1] for g in major_minor_combs]
            
            for j,line in enumerate(vcf):
                line = line.strip()
                for idx,value in enumerate(line.split('\t')[:9]):
                    self.snp_info[j,idx] = value
                genotypes = line.split('\t')[9:]
                for i,gt in enumerate(genotypes):
                    if gt == '0|0' or gt == '0/0':
                        self.data[i,j] = 0
                    elif gt in major_minor_combs:
                        self.data[i,j] = 1
                    elif gt in minor_allele_combs:
                        self.data[i,j] = 2 
                    elif gt == '.|.' or gt == './.':
                        self.data[i,j] = -1

    def sort(self, by='ID'):
        col = self.info_cols.index(by)
        idx = np.argsort(self.snp_info[:,col])
        self.data = self.data[:,idx]
        self.snp_info = self.snp_info[idx,:]
        if hasattr(self,'multi_allelic'):
            self.multi_allelic[idx]
        
    def get_multiallelic(self):
        self.multi_allelic = np.array([True if ',' in allele else False 
                                       for allele in self.snp_info[:,4]])
    def filter_multiallelic(self):
        if not hasattr(self,'multi_allelic'):
            self.get_multiallelic
        self.data = self.data[:,~self.multi_allelic]
        self.snp_info = self.snp_info[~self.multi_allelic,:]