说明：
0:
temp_5.json 文件对应的是'segment-1943605865180232897_680_000_700_000' 视频段中'000081.jpg' 到 '000085.jpg'的ground truth信息；
1:
实验的时候用  1233_later_pair_0.npy 这个文件，它是从 'segment-1943605865180232897_680_000_700_000' 视频段中的 '000082.jpg' 到 '000083.jpg'
进行warp的flow。 根据这个应该可以将 '000083.jpg' 上的reppoints warp到 '000082.jpg'上；

2：
flow保存到 npy 文件中了，
1233_former_pair_0.npy 对应 从  '000082.jpg' 到 '000081.jpg' 的光流；
1233_later_pair_0.npy 对应 从  '000082.jpg' 到 '000083.jpg' 的光流；( 你应该用这个)

3：所有的pkl文件记录的是 '000081.jpg' 到 '000085.jpg' 这5个文件对应的信息，index=1 对应 '000082.jpg'；index=2 对应 '000083.jpg'；

4：cls_feat_5_all.pkl 对应保存 forward中的 cls_feat;
cls_out_5_all.pkl 对应保存 forward中的 cls_out;




