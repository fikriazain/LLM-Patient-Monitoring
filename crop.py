import fitz
from multi_column import column_boxes

filename = 'nutrisi_pada_anak_dengan_penyakit_ginjal_clean.pdf'
src = fitz.open(f"data\pdfs\\{filename}")
doc = fitz.open()  # empty output PDF
header_margin=100
footer_margin=75
run =0 
for spage in src:  # for each page in input
    r = spage.rect  # input page rectangle
    #--------------------------------------------------------------------------
    # example: cut input page into 2 x 1 parts
    #--------------------------------------------------------------------------
    # d = fitz.Rect(0,0,0,0)  # starting at (0, 0)

    # r1 = r - (0, 0, r.width/2, 50)    # top left rect
    # r2 = r + (r.width/2, 0, 0, -50)    # top left rect

    # if not run:  
    #     r1 = r1 + (0, r.height/2 + 50, 0, 0)
    #     r2 = r2 + (0, r.height/2 + 50, 0, 0)
    #     run = True

    # rect_list = [r1, r2]  # put them in a list

    # for rx in rect_list:  # run thru rect list
    #     rx += d  # add the CropBox displacement
    #     page = doc.new_page(-1,  # new output page with rx dimensions
    #                        width = rx.width,
    #                        height = rx.height)
    #     page.show_pdf_page(
    #             page.rect,  # fill all new page with the image
    #             src,  # input document
    #             spage.number,  # input page number
    #             clip = rx,  # which part to use of input page
    #     )
   
    d = fitz.Rect(-2,-2,2,2)  # starting at (0, 0)
    bboxes = column_boxes(spage, footer_margin=footer_margin, header_margin=header_margin)
    #--------------------------------------------------------------------------
    # untuk 'Tatalaksana_Hemodialisis_pada_Anak_dan_Bayi_clean.pdf'
    #--------------------------------------------------------------------------
    # if run < 2:
    #     if run == 0 :
    #         bboxes.insert(0, bboxes.pop(3))
    #         run += 1
    #     elif run == 1:
    #         bboxes.insert(2, bboxes.pop(0))
    #         run += 1

    for i, rx in enumerate(bboxes):  # run thru rect list
        # print(i, rx)
        rx += d  # add the CropBox displacement
        page = doc.new_page(-1,  # new output page with rx dimensions
                           width = rx.width,
                           height = rx.height)
        page.show_pdf_page(
                page.rect,  # fill all new page with the image
                src,  # input document
                spage.number,  # input page number
                clip = rx,  # which part to use of input page
            )

# that's it, save output file
doc.save(f"data\pdfs-poster\\{filename}".replace(".pdf", "_poster.pdf"),
         garbage=3,  # eliminate duplicate objects
         deflate=True,  # compress stuff where possible
)