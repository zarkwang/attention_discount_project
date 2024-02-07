import os

def insert_tab(tab_name,destination_content,
               destination_type=None,
               output=None):

    source = tab_name + '.tex'

    with open(source, 'r') as source:
        source_content = source.read()

    if destination_type is not None:
        destination = destination_content + destination_type

        with open(destination, 'r') as destination:
            destination_content = destination.read()

    insert_location = r'% INSERT ' + tab_name

    # Find the insertion point in the destination content
    insert_index = destination_content.find(insert_location)

    # Insert the source content at the specified location
    updated_destination = (
            destination_content[:insert_index]
            + source_content
            + destination_content[insert_index:]
        )
    
    if output is None:
        return updated_destination
    else:
        with open(output, 'w') as f:
            f.write(updated_destination)


if __name__ == '__main__':

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    insert_tab(tab_name='reg_ols',
                destination_content='reg_ols_original',
                destination_type='.txt',
                output='reg_ols_tab.tex')
    
    insert_tab(tab_name='reg_rlm',
               destination_content='reg_rlm_original',
               destination_type='.txt',
               output='reg_rlm_tab.tex')
    
    insert_tab(tab_name='reg_combined',
               destination_content='reg_combined_original',
               destination_type='.txt',
               output='reg_combined_tab.tex')
    
    





