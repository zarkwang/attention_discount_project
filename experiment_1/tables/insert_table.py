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

    # tmp_baseline = insert_tab(tab_name='baseline_A',
    #                         destination_content='baseline_original',
    #                         destination_type='.txt')
    
    # insert_tab(tab_name='baseline_B',
    #            destination_content=tmp_baseline,
    #            output='baseline_tab.tex')
    
    # insert_tab(tab_name='utility_A',
    #             destination_content='utility_original_a',
    #             destination_type='.txt',
    #             output='utility_a_tab.tex')

    # insert_tab(tab_name='utility_B',
    #             destination_content='utility_original_b',
    #             destination_type='.txt',
    #             output='utility_b_tab.tex')

    insert_tab(tab_name='exp1_utility_model',
                destination_content='exp1_utility_original',
                destination_type='.txt',
                output='exp1_utility_tab.tex')
    
    insert_tab(tab_name='exp1_baseline_model',
                destination_content='exp1_baseline_original',
                destination_type='.txt',
                output='exp1_baseline_tab.tex')
    
    insert_tab(tab_name='exp1_utility_censor',
                destination_content='exp1_censor_original',
                destination_type='.txt',
                output='exp1_censor_tab.tex')
    





