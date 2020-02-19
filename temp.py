test_data_vp=[[] for i in range(10)]	
for i in range(len(vp_video)):
	if vp_video[i]['video_id'] in test_fuse:
		vpcase_index=vp_video[i]['vpcase_index']
		vpocclusion_record=vp_video[i]['vpocclusion_record']
		for j in range(1,11):
			for m in range(len(vpcase_index)):
				data_index=vpocclusion_record[j][m]
				vpcase_index=vpcase_index[j][m]
				data_t=data[data_index]
				data_t['occ_index']=vpcase_index
				data_t['data_index']=data_index
				test_data_vp[j].append(data_t)