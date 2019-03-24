"""
Dataset generation file generate and filter two dataframes

(first with raw features, second with derived features) for using in the model of prediction class node

"""

import pandas as pd
from helpers import *


def get_raw_features(tag_list, tag_text_list):
    """
    Create and full a dataframe "RawFeatures" in order to use later for "DerivedFeatures"

    """


    # Raw features not be used directly by ML algorithms applied for every html element(node) of webpage
    df_raw_features = pd.DataFrame(columns=['tag_name', 'element_id', 'class_name', 'children_names', 'children_ids', 'children_classes', 'image_alt',
                 'rect_size', 'num_child_elements', 'dom_subtree_depth', 'inner_text', 'child_text', 'ancestors_names',
                 'ancestors_ids', 'ancestors_classes', 'siblings_names', 'siblings_ids', 'siblings_classes',
                 'nearest_header', 'normalized_top', 'normalized_bottom', 'is_middle', 'is_leftmost', 'is_rightmost',
                 'num_siblings', 'distance_to_root', 'url', 'title', 'site', 'meta_description', 'doc_dom_depth'])

    # the name of the node
    df_raw_features.loc[:, 'tag_name'] = [tag.name for tag in tag_list]

    # element_id (if exists)
    df_raw_features.loc[:, 'element_id'] = [tag.get('id') if (tag.has_attr('id') and len(tag.get('id')) > 0) else 'CEML_NO_ID' for tag in tag_list]

    # class_name (if exists)
    df_raw_features.loc[:, 'class_name'] = [tag.get('class') if (tag.has_attr('class') and len(tag.get('class')) > 0) else 'CEML_NO_CLASS' for tag in tag_list]

    # image_alt
    df_raw_features.loc[:, 'image_alt'] = [tag.get('alt') if (tag.has_attr('alt') and tag.name == 'img' and len(tag.get('alt')) > 0) else 'CEML_NON_IMG_TAG' if tag.name != "img" else 'CEML_NO_ALT' for tag in tag_list]

    # content of the description <meta> element
    df_raw_features.loc[:, 'meta_description'] = [tag.get('content') if (tag.has_attr('content') and tag.name == 'meta' and tag.has_attr('name') and tag.get('name').lower() == 'description' and len(tag.get('content')) > 0) else 'CEML_NON_META_TAG' if tag.name != "meta" else 'CEML_NO_CONTENT' for tag in tag_list]

    # number of direct child node
    df_raw_features.loc[:, 'num_child_elements'] = [len(tag.find_all(recursive=False)) for tag in tag_list]

    # the inner text of the node
    df_raw_features.loc[:, 'inner_text'] = [getText(tag).strip() if (len(getText(tag)) > 0 and tag.name in tag_text_list) else 'CEML_NO_TEXT_TAG' if tag.name not in tag_text_list else 'CEML_NO_TEXT' for tag in tag_list]

    # the text of the all direct children of the node
    getChildText(df_raw_features, tag_list, tag_text_list)

    return df_raw_features


def get_derived_features(tag_list, tag_text_list):
    """
    Create and full a dataframe "DerivedFeatures" (future predictive variables)
    """

    # import dataframe 'get_raw_features'
    df_raw_features = get_raw_features(tag_list, tag_text_list)

    # range from 0 to the row count of dataframe
    list_row_count = range(0, df_raw_features.shape[0])

    # creation dataframe empty
    df_derived_features = pd.DataFrame(columns=['tag_name', 'has_children', 'inner_text_length', 'is_like_price', 'child_text_length', 'is_sib_p',
                 'is_sib_a', 'is_sib_input', 'is_desc_a', 'is_desc_nav', 'is_desc_ad', 'is_desc_comment', 'is_desc_main',
                 'is_desc_footer', 'is_desc_wrapper', 'is_desc_aside', 'is_desc_p', 'is_desc_div', 'is_desc_h',
                 'is_desc_ul', 'is_desc_table', 'is_desc_ol', 'is_desc_menu', 'contains_rights_reserved', 'contains_like',
                 'contains_share', 'is_link', 'is_thumbnail', 'has_depth_greater_2', 'all_desc_text_length', 'element_id',
                 'class_name', 'y'])

    # fill column 'tag_name' by the name of all node html
    df_derived_features.loc[:, 'tag_name'] = [tag.name for tag in tag_list]

    # fill column 'inner_text_length' by the word count of inner_text of the node
    df_derived_features.loc[:, 'inner_text_length'] = [len(df_raw_features.inner_text[i].split()) if df_raw_features.inner_text[i] not in ['CEML_NO_TEXT_TAG', 'CEML_NO_TEXT'] else 0 for i in list_row_count]

    # fill column 'child_text_length' by the word count of child_text of the node
    df_derived_features.loc[:, 'child_text_length'] = [len(df_raw_features.child_text[i].split()) if df_raw_features.child_text[i] != 'CEML_NO_TEXT' else 0 for i in list_row_count]

    # check if the node has at least one direct children and save his boolean values in column 'num_child_elements'
    df_derived_features.loc[:, 'has_children'] = [1 if df_raw_features.num_child_elements[i] > 0 else 0 for i in list_row_count]

    # check if the inner text of the node contains same text like 'rights reserved','like' or 'share' and save his boolean values in each column
    for elem_contains in ["rights reserved", "like", "share"]:
        get_contains_column(df_derived_features, df_raw_features, list_row_count, elem_contains)

    # 'boolean' check if node of the parent is <a> node
    df_derived_features.loc[:, 'is_desc_a'] = [1 if tag.parent.name == 'a' else 0 for tag in tag_list]

    # 'boolean' check if inner text contains the number
    df_derived_features.loc[:, 'is_like_price'] = [1 if bool(re.search(r'(\d{1,}) ?(\d{2,3}) ?(\d{3,})?', df_raw_features.loc[:, 'inner_text'][i])) else 0 for i in list_row_count]

    # text of all descendant of the current node
    df_derived_features.loc[:, 'all_desc_text_length'] = [len(''.join(tag.find_all(text=True)).strip().split()) for tag in tag_list]

    df_derived_features.loc[:, 'element_id'] = df_raw_features.loc[:, 'element_id']
    df_derived_features.loc[:, 'class_name'] = df_raw_features.loc[:, 'class_name']

    get_bool_columns(tag_list, df_derived_features, list_row_count)

    return df_derived_features


def get_bool_columns(tag_list, df_derived_features, list_row_count):
    """
    Used in addition for "DerivedFeatures" in order to fill a dataframe
    """

    # regular expression pattern for searching advertisement
    regex_ad = r'(\bad-)|(\bad_)|(\badv-)|(\badv_)|(advert)|(adblock)|(-ad\b)|(_ad\b)|(-adv\b)|(_adv\b)|(\bads)|(adbox)'

    # 'boolean' check if node is the only child node of <a> node (special for <img> node)
    is_thumbnail(tag_list, df_derived_features)

    # 'boolean' check if the node has the single child <a> node
    is_link(tag_list, df_derived_features)

    # 'boolean' check if the node has the sibling p
    is_sib_p(tag_list, df_derived_features)

    # 'boolean' check if the node has the sibling a
    is_sib_a(tag_list, df_derived_features)

    # 'boolean' check if the node has the sibling input
    is_sib_input(tag_list, df_derived_features)

    # 'boolean' check if one of the node's ancestors has the class or id "navigation"
    is_desc_nav(tag_list, df_derived_features)

    # 'boolean' check if one of the node's ancestors has the class or id "comment"
    is_desc_comment(tag_list, df_derived_features)

    # 'boolean' check if one of the node's ancestors has the class or id "main"
    is_desc_main(tag_list, df_derived_features)

    # 'boolean' check if one of the node's ancestors has the class or id "footer"
    is_desc_footer(tag_list, df_derived_features)

    # 'boolean' check if one of the node's ancestors has the class or id "wrapper"
    is_desc_wrapper(tag_list, df_derived_features)

    # 'boolean' check if one of the node's ancestors has the class or id "aside"
    is_desc_aside(tag_list, df_derived_features)

    # 'boolean' check if one of the node's ancestors has the class or id "advertisement"
    is_desc_ad(tag_list, df_derived_features, regex_ad)

    # 'boolean' check if the node has <p> ancestor
    is_desc_p(tag_list, df_derived_features)

    # 'boolean' check if the node has <div> ancestor
    is_desc_div(tag_list, df_derived_features)

    # 'boolean' check if the node has <ul> ancestor
    is_desc_ul(tag_list, df_derived_features)

    # 'boolean' check if the node has <table> ancestor
    is_desc_table(tag_list, df_derived_features)

    # 'boolean' check if the node has <h> ancestor
    is_desc_h(tag_list, df_derived_features)

    # 'boolean' check if the node has <ol> ancestor
    is_desc_ol(tag_list, df_derived_features)

    # 'boolean' check if the node has <menu> ancestor
    is_desc_menu(tag_list, df_derived_features)

    # get a real class element of all nodes
    get_class_y(df_derived_features, list_row_count)

    # 'boolean' check if the node has a depth greater 2
    has_depth_greater_2(tag_list, df_derived_features)

    return df_derived_features


def get_derived_features_filtered(soup):
    """
    Filtering out certain elements(these elements are considered as noise)

    """

    # list of all tag
    tag_list = soup.findAll()

    # list of textual tags
    tag_text_list = ["p", "div", "label", "tr", "th", "b", "span", "strong", "title", "td", "li", "h1", "h2", "h3", "h4", "h5", "h6", "dd", "dt", "mark", "em"]

    # list of the header tag
    tag_header_list = ["h1", "h2", "h3", "h4", "h5", "h6"]

    # import dataframe for filtering out same element
    df_derived_features = get_derived_features(tag_list, tag_text_list)

    # filters of certain element from navigation, advertisement and by others criteres
    df_derived_features = df_derived_features[~((df_derived_features.tag_name == 'p') & (
                (df_derived_features.is_desc_a == 1) | (df_derived_features.is_desc_nav == 1) | (
                    df_derived_features.is_desc_ad == 1) | (df_derived_features.is_link == 1)))]
    df_derived_features = df_derived_features[~((df_derived_features.tag_name == 'div') & (
                (df_derived_features.is_desc_a == 1) | (df_derived_features.is_desc_nav == 1) | (
                    df_derived_features.is_desc_ad == 1) | (df_derived_features.is_link == 1)))]
    df_derived_features = df_derived_features[
        ~((df_derived_features.is_desc_ad == 1) & (df_derived_features.tag_name == 'li'))]
    df_derived_features = df_derived_features[~((df_derived_features.is_desc_nav == 1) & (
                (df_derived_features.tag_name == 'th') | (df_derived_features.tag_name == 'td')))]
    df_derived_features = df_derived_features[~(
                (df_derived_features.tag_name.apply(lambda x: True if x in tag_text_list else False)) & (
                    df_derived_features.inner_text_length == 0))]
    df_derived_features = df_derived_features[~(
                (df_derived_features.tag_name.apply(lambda x: True if x in tag_header_list else False)) & (
                    (df_derived_features.is_desc_a == 1) | (df_derived_features.is_desc_nav == 1) | (
                        df_derived_features.is_desc_ad == 1) | (df_derived_features.is_link == 1)))]

    # delete rows in dataframe with the sectionning tags(tags with a depth greater 2 and contains a text)
    df_derived_features = df_derived_features[~((df_derived_features.tag_name.apply(
        lambda x: True if x in ['div', 'table', 'ol', 'ul', 'menu'] else False)) & (
                                                            df_derived_features.has_depth_greater_2 == True) & (
                                                            df_derived_features.all_desc_text_length > 0))]

    # delete rows in dataframe with the tags name <html>, <head>, <body> or <script>
    df_derived_features = df_derived_features[df_derived_features.tag_name.apply(lambda x: False if x in ['html', 'head', 'body', 'script'] else True)]

    # select columns for analyse
    df_derived_features = df_derived_features.drop(['has_depth_greater_2', 'all_desc_text_length', 'element_id', 'class_name', 'is_thumbnail'], axis=1)

    # delete all duplicate rows
    df_derived_features = df_derived_features.drop_duplicates()

    return df_derived_features