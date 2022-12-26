from django.shortcuts import render
from django.http import HttpResponse
from django.http import HttpResponseRedirect
from .forms import UrlForm
from . import imageOperations


def say_hello(request):
    return render(request, 'homepage.html')


def result(request):
    if request.method == 'GET':
        print("GET method")
        form = UrlForm(request.GET)
        print(request.GET)
        if form.is_valid():
            print("Form is valid")
            url = form.cleaned_data['url']
            imageOperations.download_photo(url)
            percentageValue = imageOperations.prediciton()
            return HttpResponse(percentageValue)

    return HttpResponse("Wrong url")
