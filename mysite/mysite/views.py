from django.shortcuts import render
from django.http import HttpResponse
from django.http import HttpResponseRedirect
from .forms import UrlForm
from . import imageOperations


def say_hello(request):
    return render(request, 'homepage.html')


def result(request):
    if request.method == 'POST':
        form = UrlForm(request.POST)
        print(request.POST)
        if form.is_valid():
            print("Form is valid")
            url = form.cleaned_data['url']

            try:
                imageOperations.download_photo(url)
            except ValueError as e:
                return render(request, 'errorPage.html')

            percentageValue = imageOperations.prediciton()
            return render(request, 'resultPage.html')

    return HttpResponse("Wrong url")
