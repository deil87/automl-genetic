
unflatten = function( array, parent, tree ){

    tree = typeof tree !== 'undefined' ? tree : [];
    parent = typeof parent !== 'undefined' ? parent : { id: 0 };

    var children = _.filter( array, function(child){ return child.parentid == parent.id; });

    if( !_.isEmpty( children )  ){
        if( parent.id == 0 ){
            tree = children;
        }else{
            parent['children'] = children
        }
        _.each( children, function( child ){ unflatten( array, child ) } );
    }

    return tree;
};