//
// Created by Benjamin Huang on 11/20/2018.
//
#ifndef CODE_NONOGRAMCOLOR_H
#define CODE_NONOGRAMCOLOR_H

#ifdef DISP
#include <ncurses.h>
#endif

enum NonogramColor : char {
    NGCOLOR_UNKNOWN = 0,
    NGCOLOR_WHITE = -1,
    NGCOLOR_BLACK = 1,
    NGCOLOR_HYP_WHITE = 3,
    NGCOLOR_HYP_BLACK = 5
};
    
__inline__ char ngramColorToChar(NonogramColor color) {
    switch (color) {
    case NGCOLOR_UNKNOWN: return ' ';
    case NGCOLOR_WHITE: return '.';
    case NGCOLOR_BLACK: return '#';
    case NGCOLOR_HYP_WHITE: return '8';
    case NGCOLOR_HYP_BLACK: return ':';
    default: return '?';
    }
}
     
__inline__ NonogramColor ngramColorToHypColor(NonogramColor color) {
    switch (color) {
    case NGCOLOR_WHITE: return NGCOLOR_HYP_WHITE;
    case NGCOLOR_BLACK: return NGCOLOR_HYP_BLACK;
    default: return color;
    }
}
     
__inline__ NonogramColor ngramHypColorToColor(NonogramColor color) {
    switch (color) {
    case NGCOLOR_HYP_WHITE: return NGCOLOR_WHITE;
    case NGCOLOR_HYP_BLACK: return NGCOLOR_BLACK;
    default: return color;
    }
} 
    
#endif //CODE_NONOGRAMCOLOR_H
