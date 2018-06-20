import csv
with open('C:/Users/Mike/Downloads/winequality-red.csv', 'r+') as csvfile:

  # Reading
  csvreader = csv.reader(csvfile, delimiter=';', quotechar='"')

  # Writing
  csvwriter = csv.writer(csvfile, delimiter=';')
  csvwriter.writerow(['Good'] * 5 + ['Bad'])

  for row in csvreader:
    print(', '.join(row))