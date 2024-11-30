


def load_jailbreak(args, model, dataset):
    if args.jailbreak == 'structure_ood':
        from jailbreak.structure_ood import structure_ood
        return structure_ood(args, model, dataset)