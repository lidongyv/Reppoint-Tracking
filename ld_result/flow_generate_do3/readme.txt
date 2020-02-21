1、这个代码是在stsn_one 的代码基础上修改的，主要修改了single_stage.py 的forward_train函数 和 config文件
2、所有的pkl文件是采用 /home/ld/RepPoints/ld_result/reppoint_do3 中的代码和第20个epoch的模型得到的
3、通过reppoints_baseline生成pkl的时候 和 通过基于stsn_one修改的train文件生成flow 的时候，需要图像的尺寸保持一致。1280 * 768，keep_scale=True,scale=1.67
4、生成pkl的时候测试的 score=0.3
5、生成flow的代码很简单，也可以放到test里面生成flow
