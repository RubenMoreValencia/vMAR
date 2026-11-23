#include "vmar/state_layout.h"
#include <math.h>
#include <stdint.h>

vmar_status vmar_state_layout_make(size_t parameters,vmar_state_kind kind,
                                   vmar_state_layout *result) {
    if(!result||!parameters||kind<VMAR_STATE_PARAMETERS||
       kind>VMAR_STATE_SECOND_MOMENT)return VMAR_ERR_ARGUMENT;
    size_t blocks=(size_t)kind;
    if(parameters>SIZE_MAX/sizeof(double)/blocks ||
       parameters>SIZE_MAX/blocks)return VMAR_ERR_DIMENSION;
    const size_t total=parameters*blocks;
    vmar_state_layout canonical={kind,parameters,0,total,total,total};
    if(blocks>=2)canonical.first_offset=parameters;
    if(blocks==3)canonical.second_offset=2*parameters;
    *result=canonical;
    return VMAR_OK;
}

vmar_status vmar_state_layout_validate(const vmar_state_layout *layout,
                                       const double *state,
                                       size_t supplied_doubles) {
    if(!layout||!state)return VMAR_ERR_ARGUMENT;
    vmar_state_layout expected;
    vmar_status st=vmar_state_layout_make(layout->parameters,
                                          layout->kind,&expected);
    if(st!=VMAR_OK)return st;
    if(layout->theta_offset!=expected.theta_offset ||
       layout->first_offset!=expected.first_offset ||
       layout->second_offset!=expected.second_offset ||
       layout->total_doubles!=expected.total_doubles ||
       supplied_doubles!=expected.total_doubles)return VMAR_ERR_DIMENSION;
    for(size_t j=0;j<supplied_doubles;++j) {
        if(!isfinite(state[j]))return VMAR_ERR_NUMERIC;
        if(layout->kind==VMAR_STATE_SECOND_MOMENT&&
           j>=layout->second_offset&&state[j]<0)return VMAR_ERR_NUMERIC;
    }
    return VMAR_OK;
}
