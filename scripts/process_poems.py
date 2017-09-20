
import urllib.request, json 

outfile = open("sonnet.txt", "w")
with urllib.request.urlopen("http://poetrydb.org/author,title/Shakespeare;Sonnet/lines.json") as url:
    data = json.loads(url.read().decode())
    # print(data)

for poem in data:
	for line in poem['lines']:
		outfile.write(line + "\n")
	outfile.write("\n")

outfile.close()
