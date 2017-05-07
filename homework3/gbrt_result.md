mlp v3 val error: 768.153017124
mlp v5 val error: 731.974512866

v3 base1975的feature:

| num estimator  | max depth  | loss | lr   | train loss | var loss | after averaging with mlp var loss | test loss |
|:--------------:|:----------:|:----:|:----:|:----------:|:--------:|:---------------------------------:|:---------:|
| 300 | 1 | L2 |  0.1 | 805.5084187 | 810.0666425 | | |
| 100 | 3 | L2 |  0.1 | 736.0975027 | 792.9 | | |
| 200 | 3 | L2 |  0.1 |  | 772.214375699 | 755.061657727 | |
| 300 | 3 | L2 |  0.1 | 658.1234342 | 760.4391235 | | |
| 400 | 3 | L2 |  0.1 | 639.9698171 | 754.1893936 | | |
| 500 | 3 | L2 |  0.1 | 625.9100637 | 753.6829506 | | |
| 200 | 4 | L2 |  0.1 | 633.1790772 | 752.0397254 | | |
| 300 | 4 | L2 |  0.1 | 604.2819866 | 740.1142087 | | |
| 400 | 4 | L2 |  0.1 | 586.0605376 | 737.2270994 | | |
| 500 | 4 | L2 |  0.1 | 572.8267043 | 738.4360572(overfit) | | |
| 200 | 5 | L2 |  0.1 | 586.2418189 | 744.0510261 | | |
| 300 | 5 | L2 |  0.1 | 560.7126995 | 738.102497  | | |
| 400 | 5 | L2 |  0.1 | 539.807667  | 735.459869 | 728.54336538 | after averaging: 754.005358447316 |
| 500 | 5 | L2 |  0.1 | 524.4835535 | 735.4228746 | (avergage with mlp v5) 707.537683453 | |
| 500 | 5 | huber |  0.1 | 764.2678568 | 791.9986535 | | |
| 600 | 5 | L2 |  0.1 | 510.8070939 | 735.3192345 | (avergage with mlp v5) 706.825166485 | |
| 600 | 5 | L2 |  0.05 | 557.7713573 | 737.8990287 | | |
| 600 | 6 | L2 |  0.1 | 463.4466236 | 744.2789806(overfit) | | |
| 800 | 5 | L2 |  0.1 | 510.8070939 | 737.8138925(overfit) | | |
| 800 | 5 | L2 |  0.05 | 538.432558 | 737.1049429 | | |
| 1000 | 5 | L2 |  0.01 | 630.7459094 | 753.5998915 | | |
| 1000 | 5 | L2 |  0.05 | 523.4593153 | 734.4511963 | | |
| 1200 | 5 | L2 |  0.05 | 509.1833993 | 734.1064243  | | |

v5 feature:
| num estimator  | max depth  | loss | lr   | train loss | var loss | after averaging with mlp var loss | test loss |
|:--------------:|:----------:|:----:|:----:|:----------:|:--------:|:---------------------------------:|:---------:|
| 500 | 3 | L2 |  0.1 | 603.8671054 | 732.8576197  | (avergage with mlp v5) | 716.060908129 |
| 500 | 5 | L2 |  0.1 | 509.7350028 | 729.6700776 | (avergage with mlp v5) 709.281704553 | |
反而没有上面的互补性了= =至少在validation上不过应该都差不多

v6 feature: 加入了conference(有一些教材什么... 平均值太高了... 反而因为数据不全顶会平均值不怎么样... 感觉应该用经常在该会上发表文章的人的target 引用量来试图评价...  不然确实对mlp不好做), mlp没有进步. gbrt有进步. 但是和v5 average之后还是没有前面的706好但是差不多= =
| 500 | 5 | L2 |  0.1 | 500.4015134 | 726.6880075 | 707.501934989 | |

v7 feature: 加入了合作者平均值(确实有一些outlier, 对于直接mlp regression确实不好做), 又是mlp没有进步, gbrt有进步. 而且进步还不错... 的确合作者关系是一个不错可引入的feature
| 500 | 5 | L2 |  0.1 | 463.1759192 | 708.9330943 | (avergage with mlp v5) 690.621661805 | |


| methods   | linear regression | MLP             |     GBRT      |
|:---------:|:-----------------:|:---------------:|:-------------:|
| v3 train  | 779.7379478       | 806.0648625     | 510.8070939(600, 5)   |
| v3 val    | 784.4635786       | 768.153017124   | 735.3192345(600, 5)   |
| v5 train  | 739.5785371       | 750.191589355   | 509.7350028(500, 5)   |
| v5 val    | 739.0159671       | 731.974512866   | 729.6700776(500, 5)   |
| v7 train  | 726.92577         |    | 463.1759192(500, 5)   |
| v7 val    | 729.6102374       |    | 708.9330943(500, 5)   |
| v9 train  | 725.3341671       |    | |
| v9 val    | 713.8627071(after post: 712.981110645; average with v5 mlp 711.336632437, average with gbrt v7 679.864093365. 哇我把这个交了... test 661 效果不错!)      |    |  655.4319148(600, 5) |
| v10 train | 726.3335084       |    | |
| v10 val   | 715.7601487       |    | |
| v11 train | 679.5615549       |  683.2101845(400) 550.99822998(1000)(有一定的overfit, L2) | 398.1649887(500, 5) |
| v11 val   | 699.0979139(after post:  697.798612525; average with gbrt v7 643.139887016(test 635.99225. v12新的test feature: 635.011816376182). average with gbrt v9: 635.709950679(test 643.34587298387过拟合了); )      | 724.3303373(400) 666.91595459(1000)(average with gbrtv7: 628.489951417. test:  613.985499016759)  |  636.1306552(500, 5) |
| v13 train | 679.5615549       |    | |
| v13 val   | 695.0319592(after post: 693.724912498; average with gbrt v7:)       |    | |
v14, v15用到了val的信息帮助train分每个paper的比重, 可能利用这个信息也会过拟合?. 干脆还是只用train的cut. 诶不对a...没有用... val的`total_cts_count`只被用在了test.all_papers_cts的更新上. 没有用在train.all_papers_cts或者val的更新上... 不过把val的target用在了其一些paper的最小值的限定上了 = =有道理诶. val还是不用于限定paper的cts了. 但是val的total_cts_count可以用在test的all_papers_cts的更新上
| v14 train | 679.5231961       |    |  |
| v14 val   | 684.1683708(after post: 682.756776868; average with gbrt v7: 638.041958655;)       |    | 624.0462469(500, 5) |
| v15 train | 664.5638166       |    | |
| v15 val   | 645.4476259(after post:  643.944785454; average with gbrt v7:  618.882995168. test: 638.359091555661更差了= =看来val并不能很好的反应test集的情况呀)       |    | |
ev9: python extract_feature.py -- train_features_v9.pkl.train -f all_paper_count -f num_collabrator_by_year -f citation_count_by_year_noorder  -f h_index -f neighbor_citation


v10和v9差不多:可以看到= = h-index是反作用= =基本上就是all_paper_count一个特别大...
[  3.79084195e+13   1.76894531e+02   2.75937500e+01   2.41796875e+01
  -3.30390625e+01  -8.44482422e+01   3.62280273e+01  -1.80922852e+01
  -2.12207031e+01  -1.94042969e+01   5.88964844e+00  -1.15644531e+01
  -5.71289062e+00  -2.16052246e+01   1.04550781e+01   3.02351074e+01
  -1.57399902e+01  -2.21704102e+00  -2.45029297e+01  -1.52073975e+01
  -2.15539551e+00  -3.19044189e+01   1.28862305e+01  -5.26489258e-01
  -1.04034424e+01   1.08536377e+01   1.36003418e+01   9.97961426e+00
  -1.33482666e+01  -7.48071289e+00   1.87133789e-01   3.71203613e+00
  -1.81396484e-01  -1.82781982e+00   9.81005859e+00  -5.14898682e+00
   1.04401855e+01   1.73336182e+01   4.05847168e+00   1.04727783e+01
   4.92919922e+00   3.66979980e+00   1.46357422e+01   3.87237549e+00
   1.77656250e+01   9.07531738e+00  -1.89811295e+02  -1.30491943e+02
  -8.15045166e+01  -2.44444641e+02  -1.13310303e+02  -1.56577332e+02
  -2.97419434e+01  -1.09273193e+02  -8.48739014e+01  -1.30089661e+02
  -3.55200195e+01  -1.96646851e+02  -3.78214722e+01  -1.24414185e+02
  -1.64750122e+02  -7.94532471e+01  -1.05767944e+02  -3.25397949e+01
  -3.70213623e+01  -8.59202881e+01  -7.34215088e+01  -1.28014160e+02
  -1.28521118e+02  -9.49302979e+01  -1.49711426e+02  -1.53940430e+02
  -1.40363037e+02  -6.83776855e+01  -1.12967041e+02  -1.18175293e+02
  -1.43588379e+02  -1.04003906e+02  -1.17513306e+02  -1.25501831e+02
  -9.54631348e+01  -1.14533081e+02  -1.44451416e+02  -1.34591919e+02
  -1.63880737e+02  -1.37176758e+02  -1.20513916e+02  -1.38203369e+02
  -1.26381836e+02  -1.38208008e+02  -9.60900879e+01   1.78176575e+01
  -1.37294846e+01   1.30760956e+01   2.76323318e+00   1.18852531e+02
  -6.31111145e-01   3.70954132e+00   7.15570831e+00   3.66773987e+00
   8.39960480e+00   2.64030457e+00   1.79804077e+01   1.86141968e+00
   1.01592522e+01   1.23149338e+01   9.86908340e+00   9.43890381e+00
   1.28877716e+01   2.79187012e+00   1.06852112e+01   8.44244385e+00
   1.34042053e+01   1.71810074e+01   1.20775757e+01   7.10610962e+00
   1.10029907e+01   6.37763977e+00   1.21592102e+01   1.28451080e+01
   9.73785400e+00   1.14241829e+01   1.20742798e+01   1.27956543e+01
   9.19219971e+00   1.15553131e+01   1.22512817e+01   1.37836151e+01
   1.34351959e+01   1.42512817e+01   1.17726440e+01   7.93835449e+00
   2.00711670e+01   1.82167358e+01   2.93621826e+01   3.43370972e+01
  -3.79084195e+13   2.04004622e+00]
-37.2557651066

neighbor还是有用的, 比如v10第9不准的: 216859 Michael Minkov. 在paper.txt只有一篇2010的文章

No.8 216859: res -12.1542026066 target 28111
这个在v9里能更好一点的预测:
216859  8400.19378451 28111
另外两个合作者:
104042  Gert Jan Hofstede       28421 还有一系列文章
1210435 Geert Hofstede  28113 也就这一篇文章

可以看到基本他们三个的引用都来自这一篇paper, 要怎么通过Gert其他paper的引用很低或者Gert 其他paper的合作者的citation_train很低得到更多的信息呢?

希望val能达到<630
就可以交一次

cross vali试一次

No.9 654446: res 1891.42634139 target 25809
原来的neighbor feature: array([ 673.03530856])
改变后: 19526.920760773544


v11:
foxfi@foxfi-eva6:~/homework/adavance-ml/homeworks/homework3/code$ python test_between_pkl_file.py val_features_targets_indexes_v3_base-1975.pkl sklearn_lr/results/v11.txt.val.average_gbrt.post
error:  643.139887016
最不准确的:
No.0 536438: res 2476.0 target 65380
No.1 57357: res 2039.0 target 50038
No.2 440643: res 14260.0 target 60080
No.3 403846: res 14767.0 target 60474
No.4 875530: res 9898.0 target 37902
No.5 391051: res 5514.0 target 30647
No.6 664244: res 17502.0 target 41887
No.7 507904: res 21473.0 target 2 # v14解决了
No.8 621462: res 15734.0 target 36459
No.9 696448: res 15488.0 target 36044 

foxfi@foxfi-eva6:~/homework/adavance-ml/homeworks/homework3/code$ python test_between_pkl_file.py val_features_targets_indexes_v3_base-1975.pkl sklearn_lr/results/v11.txt.val.post
error:  697.798612525
最不准确的:
No.0 536438: res 2638.0 target 65380
No.1 57357: res 2210.0 target 50038
No.2 440643: res 13550.0 target 60080
No.3 507904: res 41879.0 target 2
No.4 403846: res 19382.0 target 60474
No.5 123761: res 36456.0 target 165
No.6 1238568: res 30059.0 target 15
No.7 32265: res 30059.0 target 15
No.8 875530: res 9075.0 target 37902
No.9 870061: res 43199.0 target 16943


v11: ipython extract_feature.py -- train_features_v11.pkl -f all_paper_count -f num_collabrator_by_year -f citation_coun
t_by_year_noorder -f h_index -f neighbor_citation -f neighbor_cts_citation

v12: cts_use_val
v13: cts_valtest_init: ipython extract_feature.py -- train_features_v13.pkl -f all_paper_count -f num_collabrator_by_year -f citation_count_by_year_noorder -f h_index -f neighbor_citation -f neighbor_cts_citation --neighbor-cts-fname cts_feat_valtest_init.pkl
v14: cts_cut_use_val  python ch_final_feature.py train_features_v11.pkl cts_feat_cut_use_val.pkl train_features_v14.pkl.    
RUNNING: v15:  python ch_final_feature.py train_features_v11.pkl cts_feat_cut_use_val2.pkl train_features_v15.pkl



foxfi@foxfi-eva6:~/homework/adavance-ml/homeworks/homework3/code$ python test_between_pkl_file.py val_features_targets_indexes_v3_base-1975.pkl sklearn_lr/results/v13.txt.val.post
error:  693.724912498
最不准确的:
No.0 536438: res 2585.0 target 65380
No.1 57357: res 2124.0 target 50038
No.2 440643: res 13467.0 target 60080
No.3 507904: res 41879.0 target 2 # 这个确实解决了
No.4 403846: res 19369.0 target 60474
No.5 123761: res 34999.0 target 165
No.6 1238568: res 30059.0 target 15
No.7 32265: res 30059.0 target 15
No.8 875530: res 9053.0 target 37902
No.9 194930: res 61607.0 target 85576


cut
truncated ind: 191132; ori: 6654; cut to 0.0
#index53e9978db7602d9701f50643这个index有好多篇文章啊= =
比如这个人就是因为这个原因6654被cut到0了因为没有文章
突然发现可能会因为重index的原因... 引用错paper诶...要不要把这些数据去掉?
这些文章好多叫Introduction....真是搞笑...





训练的时候加入L1 loss应该会更好. feature比较总感觉parameter可能也需要稀疏


v14 lr:
saving model to sklearn_lr/models/v14.pkl
Writing train results to sklearn_lr/results/v14.txt.train
training error:  461751.773988
Writing val results to sklearn_lr/results/v14.txt.val
val error:  468086.359664



foxfi@foxfi-eva6:~/homework/adavance-ml/homeworks/homework3/code$ python test_between_pkl_file.py val_features_targets_indexes_v3_base-1975.pkl sklearn_lr/results/v14.txt.val.post
error:  682.756776868
最不准确的:
No.0 536438: res 2653.0 target 65380
No.1 57357: res 2463.0 target 50038
No.2 440643: res 13553.0 target 60080
No.3 403846: res 19730.0 target 60474
No.4 1238568: res 31470.0 target 15
No.5 32265: res 31470.0 target 15
No.6 875530: res 9123.0 target 37902
No.7 870061: res 43404.0 target 16943
No.8 194930: res 61361.0 target 85576
No.9 696448: res 12321.0 target 36044


foxfi@foxfi-eva6:~/homework/adavance-ml/homeworks/homework3/code$ python test_between_pkl_file.py val_features_targets_indexes_v3_base-1975.pkl sklearn_lr/results/v14.txt.val.average_gbrtv7.post
error:  638.041958655
最不准确的:
No.0 536438: res 2484.0 target 65380
No.1 57357: res 2165.0 target 50038
No.2 440643: res 14261.0 target 60080
No.3 403846: res 14941.0 target 60474
No.4 875530: res 9922.0 target 37902
No.5 391051: res 5843.0 target 30647
No.6 664244: res 17504.0 target 41887
No.7 696448: res 14589.0 target 36044
No.8 621462: res 15736.0 target 36459
No.9 1037883: res 8353.0 target 28516

v15:
foxfi@foxfi-eva6:~/homework/adavance-ml/homeworks/homework3/code$ python test_between_pkl_file.py val_features_targets_indexes_v3_base-1975.pkl sklearn_lr/results/v15.txt.val.post
error:  643.944785454
最不准确的:
No.0 536438: res 2688.0 target 65380
No.1 57357: res 2482.0 target 50038
No.2 440643: res 13487.0 target 60080
No.3 403846: res 20589.0 target 60474
No.4 875530: res 9173.0 target 37902
No.5 870061: res 44810.0 target 16943
No.6 194930: res 60550.0 target 85576
No.7 97514: res 56963.0 target 32383
No.8 183806: res 76629.0 target 54455
No.9 664244: res 20993.0 target 41887



foxfi@foxfi-eva6:~/homework/adavance-ml/homeworks/homework3/code$ python test_between_pkl_file.py val_features_targets_indexes_v3_base-1975.pkl sklearn_lr/results/v15.txt.val.average_gbrtv7.post
error:  618.882995168
最不准确的:
No.0 536438: res 2501.0 target 65380
No.1 57357: res 2175.0 target 50038
No.2 440643: res 14228.0 target 60080
No.3 403846: res 15371.0 target 60474
No.4 875530: res 9947.0 target 37902
No.5 664244: res 17490.0 target 41887
No.6 391051: res 7295.0 target 30647
No.7 621462: res 15722.0 target 36459
No.8 1037883: res 8436.0 target 28516
No.9 696448: res 16076.0 target 36044

v11: mlp: $ CUDA_VISIBLE_DEVICES="2,3" ipython tf_iid_regression.py -- --optimizer rmsprop --data_file train_features_v11.pkl.train --model simple_mlp --save_model_dir models/v11/ --save_model_file mlp_rmsprop_run1000 --run_var mlp_rmsprop_run1000 --log_dir logs/v11/ --learning_rate 0.1 --max_iter 1000
train loss: 303599.03125, sqrt: 550.99822998; val loss: 444776.90625, sqrt: 666.91595459
train_predict will be written to train_res_mlp_rmsprop_run1000.txt
val will be written to val_res_mlp_rmsprop_run1000.txt
test_predict will be written to test_res_mlp_rmsprop_run1000.txt

error:  666.882508877
最不准确的:
No.0 536438: res 2889.0 target 65380
No.1 57357: res 1302.0 target 50038
No.2 440643: res 15073.0 target 60080
No.3 507904: res 35293.0 target 2
No.4 123761: res 31496.0 target 165
No.5 403846: res 31216.0 target 60474
No.6 875530: res 10464.0 target 37902
No.7 32265: res 26879.0 target 15
No.8 1238568: res 26879.0 target 15
No.9 391051: res 7839.0 target 30647