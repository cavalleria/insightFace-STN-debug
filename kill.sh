#! /bin/bash
## File Name: kill.sh
## Author: YangAonan
## Mail: Yang.Aonan@intellifusion.com
## Copyright (c) 2018-2019 IntellIfusion.com, Inc. All Rights Reserved
## Created Time: Thu Nov 21 07:17:10 2019


ps -ef | grep "train_parall *" | grep -v grep | cut -c 9-15 | xargs kill -9


##* vim: set ts=4 sw=4 sts=4 tw=100 ##
