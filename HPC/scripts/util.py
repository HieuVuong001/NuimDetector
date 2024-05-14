def get_dict(textfile):
	mapper = {}
	with open(textfile, 'r') as infile:
		lines = infile.readlines()
		for line in lines:
			key, value = line.split(':')
			key, value = key.strip(), value.strip()
			
			mapper[value] = int(key)

	return mapper
