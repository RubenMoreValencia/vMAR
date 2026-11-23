#include "vmar/state_layout.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>

static int checks,failures;
#define CHECK(c) do{++checks;if(!(c)){++failures;fprintf(stderr,"FAIL %d: %s\n",__LINE__,#c);}}while(0)

int main(void) {
    for(int kind=1;kind<=3;++kind) {
        vmar_state_layout layout={0};
        CHECK(vmar_state_layout_make(4,(vmar_state_kind)kind,&layout)==VMAR_OK);
        CHECK(layout.theta_offset==0);
        CHECK(layout.total_doubles==(size_t)kind*4);
        CHECK(layout.first_offset==(kind>=2?4u:layout.total_doubles));
        CHECK(layout.second_offset==(kind==3?8u:layout.total_doubles));
        double state[12]={0};
        CHECK(vmar_state_layout_validate(&layout,state,layout.total_doubles)==VMAR_OK);
        CHECK(vmar_state_layout_validate(&layout,state,layout.total_doubles-1)==VMAR_ERR_DIMENSION);
        state[0]=NAN;
        CHECK(vmar_state_layout_validate(&layout,state,layout.total_doubles)==VMAR_ERR_NUMERIC);
        state[0]=0;
        if(kind==3) {
            state[8]=-0.1;
            CHECK(vmar_state_layout_validate(&layout,state,12)==VMAR_ERR_NUMERIC);
            state[8]=0;
        }
        layout.first_offset++;
        CHECK(vmar_state_layout_validate(&layout,state,(size_t)kind*4)==VMAR_ERR_DIMENSION);
    }
    vmar_state_layout layout={0};
    CHECK(vmar_state_layout_make(0,VMAR_STATE_PARAMETERS,&layout)==VMAR_ERR_ARGUMENT);
    CHECK(vmar_state_layout_make(3,(vmar_state_kind)42,&layout)==VMAR_ERR_ARGUMENT);
    CHECK(vmar_state_layout_make(SIZE_MAX,VMAR_STATE_SECOND_MOMENT,&layout)==VMAR_ERR_DIMENSION);
    CHECK(vmar_state_layout_make(3,VMAR_STATE_FIRST_MEMORY,NULL)==VMAR_ERR_ARGUMENT);
    printf("vMAR unified state layout contracts: %d assertions, %d failures\n",checks,failures);
    return failures!=0;
}
