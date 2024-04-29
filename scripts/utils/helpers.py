def time_keeper(elapsed):
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    return '{:0>2}h: {:0>2}m: {:05.2f}s'.format(int(hours), int(minutes), seconds)

