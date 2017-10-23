#!/usr/bin/env python
import sys

# reduce the output from mapper, just do the counting
current_key = None
current_count = 0
key = None
unique_word = 0


for line in sys.stdin:
    line = line.strip()
    key, num = line.split("\t", 1)
    try:
        count = int(num)
    except ValueError:
        continue
    if current_key == key:
        current_count += count
    else:
        if current_key:
            if current_key.startswith('Y='):
                print '%s\t%s' %(current_key, current_count)
#else:
#unique_word += 1
        current_key = key
        current_count = count

# count the last input line
if current_key == key:
    if current_key.startswith('Y='):
        print '%s\t%s' % (current_key, current_count)
#else:
#unique_word += 1
#print the count of the unique word
#print 'W=*\t%s' % (unique_word)


