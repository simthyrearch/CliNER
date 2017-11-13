######################################################################
#  CliNER - error.py                                                 #
#                                                                    #
#  Kevin Wacome                                   kwacome@cs.uml.edu #
#                                                                    #
#  Purpose: Evaluate predictions of concept labels against gold.     #
######################################################################


__author__ = 'Willie Boag'
__date__   = 'Dec. 26, 2015'



import os
import sys
import argparse
import glob
import tools
from copy import deepcopy

from documents import Document
from documents import labels



def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(prog='cliner evaluate')
    parser.add_argument("--txt",
        dest = "txt",
        help = "Glob of .txt files of discharge summaries",
    )
    parser.add_argument("--predictions",
        dest = "pred",
        help = "Directory where predictions  are stored.",
    )
    parser.add_argument("--gold",
        dest = "gold",
        help = "Directory where gold standard is stored.",
    )
    parser.add_argument("--format",
        dest = "format",
        help = "Data format ( con )"
    )
    parser.add_argument("--output",
        dest = "output",
        help = "Write the evaluation to a file rather than STDOUT",
    )
    args = parser.parse_args()


    if not args.txt:
        print '\n\tERROR: must provide --txt argument\n'
        parser.print_help(sys.stderr)
        print >>sys.stderr,  ''
        exit(1)

    if not args.pred:
        print '\n\tERROR: must provide --pred argument\n'
        parser.print_help(sys.stderr)
        print >>sys.stderr,  ''
        exit(1)

    if not args.gold:
        print '\n\tERROR: must provide --gold argument\n'
        parser.print_help(sys.stderr)
        print >>sys.stderr,  ''
        exit(1)


    if args.format:
        format = args.format
    else:
        print '\n\tERROR: must provide --format argument\n'
        parser.print_help(sys.stderr)
        print >>sys.stderr,  ''
        exit(1)



    # Is output destination specified?
    if args.output:
        args.output = open(args.output, "w")
    else:
        args.output = sys.stdout


    # Must specify output format
    if format not in ['i2b2']:
        print >>sys.stderr, '\n\tError: Must specify output format'
        print >>sys.stderr,   '\tAvailable formats: i2b2'
        print >>sys.stderr, ''
        parser.print_help(sys.stderr)
        print >>sys.stderr,  ''
        exit(1)


    # List of medical text
    txt_files = glob.glob(args.txt)
    txt_files_map = tools.map_files(txt_files)
    wildcard = '*.con'

    # List of gold data
    ref_files = glob.glob( os.path.join(args.gold, wildcard) )
    ref_files_map = tools.map_files(ref_files)

    # List of predictions
    pred_files = glob.glob( os.path.join(args.pred, wildcard) )
    pred_files_map = tools.map_files(pred_files)

    # Grouping of text, predictions, gold
    files = []
    for k in txt_files_map:
        if k in pred_files_map and k in ref_files_map:
            files.append((txt_files_map[k], pred_files_map[k], ref_files_map[k]))


    # txt          <- medical text
    # annotations  <- predictions
    # gold         <- gold standard

    if len(files) == 0:
        print "No files to be evaluated"
        exit()

    print
    for txt, annotations, gold in files:

        # Read predictions and gols standard data
        cnote = Document(txt, annotations)
        rnote = Document(txt,        gold)

        sents = rnote.getTokenizedSentences()

        # Note - can also get first pass (IOB labels)
        ref  = rnote.conlist()
        pred = cnote.conlist()

        for i,toks,pline,rline in zip(range(len(sents)),sents,pred,ref):
            for j,token,rlab,plab in zip(range(len(pline)),toks,rline,pline):
                if rlab != plab:
                    ind = max(0, j-3)
                    #print 'ref:  ', rline[j-3:j+3]
                    #print 'pred: ', pline[j-3:j+3]
                    print token
                    for k in range(ind,j):
                        print ' ' * (len(toks[k]) + 4),
                    print '<>'
                    print toks[j-3:j+3]
                    print '\tpred: ', plab
                    print '\tref:  ', rlab
                    print '\n'



if __name__ == '__main__':
    main()
