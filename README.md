# Introduction
There are 11 detection methods and 6 adaptation methods:
Detection methods are:D3, IKSbdd, ITA, Kruskal, LD3, OCDD, idddsda, ks_2samp, kswin, mannwhitneyu, ttest.
adaptation methods are:LWF, client_clustering, learnandepoch, reset, client_driftchose, client_weight.

Including [libcdd](https://github.com/Shuo-Wang84/FL-VirtualConceptDrift/tree/main/libcdd) comes from https://github.com/HsiangYangChu/LIBCDD/tree/master.

# Usage
Install python dependencies.
```javascript
conda create --name FLvirtual python=3.8 -y
conda activate FLvirtual

pip install -r requirement.txt
```

# Run
Select the appropriate detection and adaptation code in [Fl_virtualStream](https://github.com/Shuo-Wang84/FL-VirtualConceptDrift/tree/main/FL_VirtualStream) to run.




