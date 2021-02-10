def arr(args):
    if args.dataset == 'Beidian':
        args.num_user = 3773
        args.num_item = 4544
    elif args.dataset == 'Beibei':
        args.num_user = 149361
        args.num_item = 30486
    return args
